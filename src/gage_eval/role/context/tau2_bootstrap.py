"""Tau2 bootstrap context provider (initialize env/user/tools)."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.registry import registry


@registry.asset(
    "context_impls",
    "tau2_bootstrap",
    desc="Tau2 bootstrap context provider (init env/user/tools)",
    tags=("tau2", "context"),
)
class Tau2BootstrapContext:
    """Initialize Tau2 runtime state and seed initial messages."""

    def provide(self, payload: Dict[str, Any], _state=None) -> Dict[str, Any]:
        # STEP 1: Resolve Tau2 runtime.
        sample = payload.get("sample") or {}
        runtime = _resolve_tau2_runtime(payload)
        initializer = getattr(runtime, "initialize_task", None)
        if not callable(initializer):
            raise RuntimeError("tau2_bootstrap requires a runtime with initialize_task")

        # STEP 2: Initialize task state and inject tools/messages.
        result = initializer(sample)
        if not isinstance(result, dict):
            return {}
        output: Dict[str, Any] = {}
        tools_schema = result.get("tools_schema")
        if isinstance(tools_schema, list):
            output["tools_schema"] = tools_schema
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            output["metadata"] = metadata
        return output


def _resolve_tau2_runtime(payload: Dict[str, Any]) -> Any:
    environment_lease = payload.get("environment_lease")
    runtime = getattr(environment_lease, "environment", None)
    if runtime is not None:
        return runtime
    provider = payload.get("sandbox_provider")
    get_handle = getattr(provider, "get_handle", None)
    if callable(get_handle):
        handle = get_handle()
        runtime = getattr(handle, "sandbox", None) if handle is not None else None
        if runtime is not None:
            return runtime
    raise ValueError("tau2_bootstrap requires environment_lease")
