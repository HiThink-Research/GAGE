from __future__ import annotations

import json
from typing import Any, Mapping

from gage_eval.agent_eval_kits.common import resolve_sample_artifact_target
from gage_eval.utils.benchmark_helpers.tau2 import resolve_tau2_termination_reason


def persist_tau2_artifacts(
    *,
    session: Any,
    scheduler_output: Mapping[str, Any] | None,
    sandbox_provider: Any = None,
) -> dict[str, str]:
    """Persist Tau2 benchmark-owned artifacts for one sample.

    Args:
        session: Runtime session owning the sample artifact layout.
        scheduler_output: Raw scheduler output emitted by the workflow scheduler.
        sandbox_provider: Optional runtime provider exposing Tau2 state access.

    Returns:
        A mapping of artifact ids to sample-root-relative paths.
    """

    output = dict(scheduler_output or {})
    runtime_state = _normalize_runtime_state(
        _capture_tau2_state(sandbox_provider),
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


def _capture_tau2_state(sandbox_provider: Any) -> dict[str, Any]:
    if sandbox_provider is None:
        return {}
    handle = sandbox_provider.get_handle()
    runtime = handle.sandbox if handle is not None else None
    getter = getattr(runtime, "get_state", None)
    if not callable(getter):
        return {}
    state = getter()
    return dict(state) if isinstance(state, dict) else {}


def _build_trajectory_payload(
    scheduler_output: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
) -> dict[str, Any]:
    agent_trace = scheduler_output.get("agent_trace")
    if isinstance(agent_trace, list) and agent_trace:
        return {
            "source": "agent_trace",
            "events": agent_trace,
        }
    messages = runtime_state.get("messages")
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


def _write_json_artifact(session: Any, filename: str, payload: Any) -> str:
    target, relative_path = resolve_sample_artifact_target(session, filename)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return relative_path
