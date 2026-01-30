"""Tau2 bootstrap context provider (initialize env/user/tools)."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.registry import registry
from gage_eval.sandbox.provider import SandboxProvider


@registry.asset(
    "context_impls",
    "tau2_bootstrap",
    desc="Tau2 bootstrap context provider (init env/user/tools)",
    tags=("tau2", "context"),
)
class Tau2BootstrapContext:
    """Initialize Tau2 runtime state and seed initial messages."""

    def provide(self, payload: Dict[str, Any], _state=None) -> Dict[str, Any]:
        # STEP 1: Resolve sandbox runtime.
        sample = payload.get("sample") or {}
        sandbox_provider = payload.get("sandbox_provider")
        if not isinstance(sandbox_provider, SandboxProvider):
            raise ValueError("tau2_bootstrap requires sandbox_provider")
        handle = sandbox_provider.get_handle()
        if handle is None or handle.sandbox is None:
            raise RuntimeError("tau2_bootstrap sandbox handle missing")
        runtime = handle.sandbox
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

