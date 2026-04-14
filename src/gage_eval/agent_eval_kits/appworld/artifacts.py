from __future__ import annotations

import json
from typing import Any

from gage_eval.agent_eval_kits.common import resolve_sample_artifact_target


def persist_appworld_artifacts(
    *,
    session: Any,
    scheduler_output: dict[str, Any] | None,
    saved_payload: dict[str, Any] | None,
) -> dict[str, str]:
    """Persist AppWorld benchmark-owned artifacts for one sample.

    Args:
        session: Sample-scoped runtime session.
        scheduler_output: Raw scheduler/client output for the sample.
        saved_payload: Runtime-owned `/save` response payload.

    Returns:
        A mapping of artifact ids to sample-root-relative artifact paths.
    """

    normalized_output = dict(scheduler_output or {})
    normalized_save = dict(saved_payload or {})
    save_outputs = _extract_save_outputs(normalized_save)
    artifact_paths: dict[str, str] = {}

    # STEP 1: Persist the raw AppWorld save payload.
    artifact_paths["appworld_save"] = _persist_json_artifact(
        session,
        "appworld_save.json",
        normalized_save,
    )

    # STEP 2: Persist AppWorld outputs derived from the save contract.
    artifact_paths["appworld_outputs"] = _persist_json_artifact(
        session,
        "appworld_outputs.json",
        {
            "output": save_outputs,
            "is_empty": save_outputs is None,
        },
    )

    # STEP 3: Persist benchmark-owned tool trace exported from the scheduler result.
    artifact_paths["appworld_tool_trace"] = _persist_json_artifact(
        session,
        "appworld_tool_trace.json",
        {
            "agent_trace": list(normalized_output.get("agent_trace") or []),
        },
    )

    # STEP 4: Persist execution logs needed for failure diagnosis.
    artifact_paths["appworld_logs"] = _persist_json_artifact(
        session,
        "appworld_logs.json",
        _build_log_payload(normalized_output, normalized_save, save_outputs),
    )

    return artifact_paths


def _extract_save_outputs(saved_payload: dict[str, Any]) -> Any:
    if "output" in saved_payload:
        return saved_payload.get("output")
    if "outputs" in saved_payload:
        return saved_payload.get("outputs")
    if saved_payload:
        return saved_payload
    return None


def _build_log_payload(
    scheduler_output: dict[str, Any],
    saved_payload: dict[str, Any],
    save_outputs: Any,
) -> dict[str, Any]:
    agent_trace = list(scheduler_output.get("agent_trace") or [])
    return {
        "status": scheduler_output.get("status"),
        "answer": scheduler_output.get("answer"),
        "stdout": scheduler_output.get("stdout"),
        "stderr": scheduler_output.get("stderr"),
        "error": scheduler_output.get("error"),
        "final_response": scheduler_output.get("final_response"),
        "tool_trace_step_count": len(agent_trace),
        "tool_trace_is_empty": len(agent_trace) == 0,
        "save_payload_has_output_key": "output" in saved_payload,
        "save_output_is_empty": save_outputs is None,
    }


def _persist_json_artifact(session: Any, filename: str, payload: Any) -> str:
    target, relative_path = resolve_sample_artifact_target(session, filename)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return relative_path
