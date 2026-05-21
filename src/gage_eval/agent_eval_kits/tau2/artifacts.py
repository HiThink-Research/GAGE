from __future__ import annotations

import json
from typing import Any, Mapping

from gage_eval.agent_eval_kits.common import resolve_sample_artifact_target
from gage_eval.agent_eval_kits.tau2._helpers import resolve_tau2_termination_reason
from gage_eval.agent_runtime.serialization import to_json_compatible
from gage_eval.reporting.privacy import SecretFilter


def persist_tau2_artifacts(
    *,
    session: Any,
    scheduler_output: Mapping[str, Any] | None,
    environment_lease: Any = None,
    sandbox_provider: Any = None,
) -> dict[str, str]:
    """Persist Tau2 benchmark-owned artifacts for one sample.

    Args:
        session: Runtime session owning the sample artifact layout.
        scheduler_output: Raw scheduler output emitted by the workflow scheduler.
        environment_lease: Optional runtime lease exposing Tau2 state access.

    Returns:
        A mapping of artifact ids to sample-root-relative paths.
    """

    output = dict(scheduler_output or {})
    runtime_state = _normalize_runtime_state(
        _capture_tau2_state(environment_lease or sandbox_provider),
        scheduler_output=output,
    )
    artifact_paths: dict[str, str] = {}

    # STEP 1: Persist the latest runtime state snapshot.
    artifact_paths["tau2_state"] = _write_json_artifact(
        session,
        "tau2_state.json",
        runtime_state,
    )

    # STEP 2: Persist the visible execution trajectory.
    artifact_paths["tau2_trajectory"] = _write_json_artifact(
        session,
        "tau2_trajectory.json",
        _build_trajectory_payload(output, runtime_state),
    )

    # STEP 3: Persist stable cost and termination diagnostics.
    artifact_paths["tau2_cost"] = _write_json_artifact(
        session,
        "tau2_cost.json",
        _build_cost_payload(output, runtime_state),
    )
    return artifact_paths


def _capture_tau2_state(runtime_source: Any) -> dict[str, Any]:
    if runtime_source is None:
        return {}
    runtime = getattr(runtime_source, "environment", None)
    if runtime is None and hasattr(runtime_source, "get_handle"):
        handle = runtime_source.get_handle()
        runtime = getattr(handle, "sandbox", None) if handle is not None else None
    if runtime is None:
        runtime = runtime_source
    getter = getattr(runtime, "get_state", None)
    if not callable(getter):
        return {}
    state = getter()
    return dict(state) if isinstance(state, dict) else {}


def _build_trajectory_payload(
    scheduler_output: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
) -> dict[str, Any]:
    messages = runtime_state.get("messages")
    if isinstance(messages, list) and messages:
        return {
            "source": "runtime_state.messages",
            "events": messages,
        }
    agent_trace = scheduler_output.get("agent_trace")
    if isinstance(agent_trace, list) and agent_trace:
        return {
            "source": "agent_trace",
            "events": agent_trace,
        }
    if isinstance(messages, list):
        return {
            "source": "runtime_state.messages",
            "events": messages,
        }
    return {
        "source": "unavailable",
        "events": [],
    }


def _build_cost_payload(
    scheduler_output: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "agent_cost": runtime_state.get("agent_cost", scheduler_output.get("agent_cost")),
        "user_cost": runtime_state.get("user_cost", scheduler_output.get("user_cost")),
        "agent_total_tokens": runtime_state.get(
            "agent_total_tokens",
            scheduler_output.get("agent_total_tokens", scheduler_output.get("total_tokens")),
        ),
        "user_total_tokens": runtime_state.get("user_total_tokens", scheduler_output.get("user_total_tokens")),
        "termination_reason": runtime_state.get(
            "termination_reason",
            scheduler_output.get("termination_reason"),
        ),
    }


def _normalize_runtime_state(
    runtime_state: Mapping[str, Any],
    *,
    scheduler_output: Mapping[str, Any],
) -> dict[str, Any]:
    """Fill stable Tau2 terminal diagnostics for artifact persistence."""

    normalized = dict(runtime_state or {})
    normalized["termination_reason"] = _normalize_termination_reason(
        runtime_state.get("termination_reason", scheduler_output.get("termination_reason"))
    )
    normalized["agent_cost"] = _coerce_cost(
        runtime_state.get("agent_cost", scheduler_output.get("agent_cost"))
    )
    normalized["user_cost"] = _coerce_cost(
        runtime_state.get("user_cost", scheduler_output.get("user_cost"))
    )
    normalized["agent_total_tokens"] = _coerce_optional_float(
        runtime_state.get(
            "agent_total_tokens",
            scheduler_output.get("agent_total_tokens", scheduler_output.get("total_tokens")),
        )
    )
    normalized["user_total_tokens"] = _coerce_optional_float(
        runtime_state.get("user_total_tokens", scheduler_output.get("user_total_tokens"))
    )
    termination_detail = runtime_state.get("termination_detail", scheduler_output.get("termination_detail"))
    if termination_detail is not None:
        normalized["termination_detail"] = str(termination_detail)
    normalized["agent_exhausted"] = bool(
        runtime_state.get("agent_exhausted")
        or normalized.get("termination_reason") == "agent_error"
        or normalized.get("termination_detail") in {"no_tool_call_from_agent", "agent_loop_max_turns"}
    )
    return normalized


def _normalize_termination_reason(value: Any) -> str:
    resolved = resolve_tau2_termination_reason(value, fallback="too_many_errors")
    raw_value = getattr(resolved, "value", resolved)
    text = str(raw_value or "").strip()
    return text or "too_many_errors"


def _coerce_cost(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coerce_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_json_artifact(session: Any, filename: str, payload: Any) -> str:
    safe_payload = _report_safe_value(payload)
    sink = _resolve_artifact_sink(session)
    if sink is not None:
        writer = getattr(sink, "write_artifact", None)
        if callable(writer):
            ref = writer(
                run_id=str(getattr(session, "run_id")),
                task_id=str(getattr(session, "task_id")),
                sample_id=str(getattr(session, "sample_id")),
                trial_id=_resolve_trial_id(session),
                owner="agent",
                name=filename,
                content=to_json_compatible(safe_payload),
                mime_type="application/json",
            )
            path = getattr(ref, "path", None)
            if isinstance(path, str) and path:
                return path

    target, relative_path = resolve_sample_artifact_target(session, filename)
    target.write_text(
        json.dumps(to_json_compatible(safe_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return relative_path


def _resolve_artifact_sink(session: Any) -> Any | None:
    runtime_context = getattr(session, "runtime_context", None)
    if isinstance(runtime_context, dict) and runtime_context.get("artifact_sink") is not None:
        sink = runtime_context.get("artifact_sink")
    else:
        sink = getattr(session, "artifact_sink", None)
    if sink is None:
        return None
    required_ids = ("run_id", "task_id", "sample_id")
    if not all(getattr(session, key, None) for key in required_ids):
        return None
    return sink


def _resolve_trial_id(session: Any) -> str:
    runtime_context = getattr(session, "runtime_context", None)
    if isinstance(runtime_context, dict) and runtime_context.get("trial_id"):
        return str(runtime_context["trial_id"])
    scheduler_state = getattr(session, "scheduler_state", None)
    if isinstance(scheduler_state, dict) and scheduler_state.get("trial_id"):
        return str(scheduler_state["trial_id"])
    return "trial_0001"


def _report_safe_value(value: Any) -> Any:
    return SecretFilter().redact(value).value
