"""Bridge runtime verifier input into Tau2 verifier requests."""

from __future__ import annotations

from typing import Any, Mapping

from gage_eval.agent_eval_kits.tau2.trace_mapping import build_tool_trace_summary


def build_tau2_verifier_request(
    *,
    sample_id: str,
    sample: Mapping[str, Any],
    scheduler_result: Mapping[str, Any],
    runtime_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the kit-owned Tau2 verifier request payload."""

    artifact_paths = scheduler_result.get("artifact_paths")
    if not isinstance(artifact_paths, Mapping):
        artifact_paths = {}
    trace_events = runtime_context.get("trace_events")
    if not isinstance(trace_events, list):
        trace_events = scheduler_result.get("trace_events") if isinstance(scheduler_result.get("trace_events"), list) else []

    return {
        "sample_id": sample_id,
        "sample": dict(sample or {}),
        "runtime_state": _resolve_runtime_state(scheduler_result, runtime_context),
        "scheduler_result": dict(scheduler_result or {}),
        "trajectory_ref": artifact_paths.get("tau2_trajectory"),
        "runtime_state_ref": artifact_paths.get("tau2_state"),
        "trace_ref": artifact_paths.get("trace"),
        "tool_trace_summary": build_tool_trace_summary(trace_events),
    }


def _resolve_runtime_state(
    scheduler_result: Mapping[str, Any],
    runtime_context: Mapping[str, Any],
) -> dict[str, Any]:
    environment_state = _runtime_state_from_environment(runtime_context)
    if environment_state:
        return environment_state

    runtime_state = scheduler_result.get("runtime_state")
    if isinstance(runtime_state, Mapping) and _looks_like_tau2_runtime_state(runtime_state):
        return dict(runtime_state)

    return {}


def _runtime_state_from_environment(runtime_context: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _resolve_runtime_source(runtime_context)
    getter = getattr(runtime, "get_state", None)
    if not callable(getter):
        return {}
    state = getter()
    return dict(state) if isinstance(state, Mapping) else {}


def _resolve_runtime_source(runtime_context: Mapping[str, Any]) -> Any | None:
    for key in ("environment_lease", "tau2_runtime", "runtime"):
        source = runtime_context.get(key)
        if source is None:
            continue
        runtime = getattr(source, "environment", None)
        return runtime or source

    provider = runtime_context.get("sandbox_provider")
    if provider is not None and hasattr(provider, "get_handle"):
        handle = provider.get_handle()
        runtime = getattr(handle, "sandbox", None) if handle is not None else None
        if runtime is not None:
            return runtime
    return None


def _looks_like_tau2_runtime_state(runtime_state: Mapping[str, Any]) -> bool:
    return any(
        key in runtime_state
        for key in (
            "task_id",
            "domain",
            "messages",
            "termination_reason",
            "termination_detail",
            "agent_cost",
            "user_cost",
            "agent_exhausted",
        )
    )
