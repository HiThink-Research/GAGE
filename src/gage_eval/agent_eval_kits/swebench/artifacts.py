from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from gage_eval.agent_eval_kits.common import resolve_sample_artifact_target
from gage_eval.agent_eval_kits.swebench.judge.failure_categories import (
    resolve_swebench_failure_category,
)
from gage_eval.agent_eval_kits.swebench.judge.patch_extraction import clean_patch_content


def persist_swebench_artifacts(
    *,
    session: Any,
    scheduler_output: Mapping[str, Any] | None,
    sandbox_provider: Any = None,
) -> dict[str, str]:
    """Persist SWE-bench runtime artifacts under the sample artifact root.

    Args:
        session: Runtime session owning the sample artifact layout.
        scheduler_output: Raw scheduler output to normalize.
        sandbox_provider: Deprecated compatibility parameter; ignored by the v2 artifact path.

    Returns:
        A mapping of artifact keys to sample-root-relative paths.
    """

    output = dict(scheduler_output or {})
    persisted: dict[str, str] = {}

    # STEP 1: Preserve already-materialized artifacts when they exist on disk.
    existing_paths = output.get("artifact_paths")
    if isinstance(existing_paths, Mapping):
        persisted.update(_materialized_artifact_paths(session, existing_paths))

    # STEP 2: Persist patch evidence from the strongest available source.
    del sandbox_provider
    patch_content = _resolve_patch_content(output)
    if patch_content:
        persisted["submission_patch"] = _write_text_artifact(
            session,
            "submission.patch",
            patch_content,
        )

    # STEP 3: Persist lightweight diagnostics so the artifact directory is auditable.
    agent_trace = output.get("agent_trace")
    trace_diagnostics = _build_trace_diagnostics(agent_trace)
    if isinstance(agent_trace, list) and agent_trace:
        persisted["agent_trace"] = _write_json_artifact(
            session,
            "agent_trace.json",
            agent_trace,
        )
    answer = output.get("answer")
    if isinstance(answer, str) and answer.strip():
        persisted["final_response"] = _write_text_artifact(
            session,
            "final_response.txt",
            answer,
        )
    persisted["swebench_diagnostics"] = _write_json_artifact(
        session,
        "swebench_diagnostics.json",
        _build_swebench_diagnostics(
            scheduler_output=output,
            submission_patch_present="submission_patch" in persisted,
            trace_diagnostics=trace_diagnostics,
        ),
    )

    return persisted


def update_swebench_diagnostics_post_verifier(
    *,
    session: Any,
    verifier_output: Mapping[str, Any] | None,
) -> None:
    """Merge SWE-bench verifier verdict fields into diagnostics after judge execution."""

    diagnostics_path = _diagnostics_path(session)
    if diagnostics_path is None:
        return

    diagnostics = _read_json_object(diagnostics_path)
    if not diagnostics:
        return

    verifier = dict(verifier_output or {})
    category_base = {
        key: value
        for key, value in diagnostics.items()
        if key not in {"failure_category", "final_failure_category"}
    }
    verifier_category = verifier.get("failure_category")
    if verifier_category == "unknown":
        verifier_category = None
        verifier = {key: value for key, value in verifier.items() if key != "failure_category"}
    merged_for_category = {
        **category_base,
        **verifier,
    }
    diagnostics.update(
        {
            "verifier_status": verifier.get("status"),
            "resolved": verifier.get("resolved"),
            "score": verifier.get("score"),
            "failure_reason": verifier.get("failure_reason"),
            "failure_code": verifier.get("failure_code"),
            "failure_category": verifier_category or resolve_swebench_failure_category(verifier),
            "patch_applied_via": verifier.get("patch_applied_via"),
            "final_failure_category": resolve_swebench_failure_category(merged_for_category),
        }
    )
    _write_json_path(diagnostics_path, diagnostics)


def _build_swebench_diagnostics(
    *,
    scheduler_output: Mapping[str, Any],
    submission_patch_present: bool,
    trace_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    agent_trace = scheduler_output.get("agent_trace")
    answer = scheduler_output.get("answer")
    trace_stats = _collect_trace_stats(agent_trace)
    failure_category_payload = {
        **dict(scheduler_output),
        "artifact_spillover_count": trace_stats["artifact_spillover_count"],
        "max_tool_output_bytes": trace_stats["max_tool_output_bytes"],
        "repeated_command_count": trace_stats["repeated_command_count"],
        "parse_error_count": trace_stats["parse_error_count"],
        "tool_protocol_error_count": trace_stats["tool_protocol_error_count"],
        "recent_errors": list((trace_diagnostics or {}).get("recent_errors") or []),
    }
    diagnostics = {
        "diagnostic_version": 2,
        "prompt_source": scheduler_output.get("prompt_source") or scheduler_output.get("source") or "unknown",
        "prompt_present": bool(scheduler_output.get("prompt") or scheduler_output.get("messages")),
        "retry_count": int(scheduler_output.get("retry_count") or trace_stats["retry_count"]),
        "retry_exhausted": bool(scheduler_output.get("retry_exhausted")),
        "max_tool_output_bytes": trace_stats["max_tool_output_bytes"],
        "artifact_spillover_count": trace_stats["artifact_spillover_count"],
        "final_failure_category": resolve_swebench_failure_category(failure_category_payload),
        "submission_patch_present": submission_patch_present,
        "agent_trace_step_count": len(agent_trace) if isinstance(agent_trace, list) else 0,
        "answer_present": isinstance(answer, str) and bool(answer.strip()),
        "required_artifact": "submission.patch",
        "missing_commands": list((trace_diagnostics or {}).get("missing_commands") or []),
        "recent_errors": list((trace_diagnostics or {}).get("recent_errors") or []),
        "failure_reason": scheduler_output.get("failure_reason"),
        "failure_code": scheduler_output.get("failure_code"),
        "tool_protocol_error_count": trace_stats["tool_protocol_error_count"],
        "parse_error_count": trace_stats["parse_error_count"],
        "repeated_command_count": trace_stats["repeated_command_count"],
    }
    return diagnostics


def _diagnostics_path(session: Any) -> Path | None:
    layout = getattr(session, "artifact_layout", {}) or {}
    artifacts_dir = layout.get("artifacts_dir")
    if isinstance(artifacts_dir, str) and artifacts_dir:
        return Path(artifacts_dir) / "swebench_diagnostics.json"
    sample_root = layout.get("sample_root")
    if isinstance(sample_root, str) and sample_root:
        return Path(sample_root) / "artifacts" / "swebench_diagnostics.json"
    return None


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json_path(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _collect_trace_stats(agent_trace: Any) -> dict[str, int]:
    stats = {
        "retry_count": 0,
        "max_tool_output_bytes": 0,
        "artifact_spillover_count": 0,
        "tool_protocol_error_count": 0,
        "parse_error_count": 0,
        "repeated_command_count": 0,
    }
    if not isinstance(agent_trace, list):
        return stats
    command_counts: dict[str, int] = {}
    for entry in agent_trace:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("retry") or entry.get("attempt"):
            stats["retry_count"] += 1
        output = entry.get("output")
        output_text = _extract_text_payload(output)
        stats["max_tool_output_bytes"] = max(
            stats["max_tool_output_bytes"],
            len(output_text.encode("utf-8")),
        )
        artifact_refs = entry.get("artifact_refs")
        if isinstance(artifact_refs, list):
            stats["artifact_spillover_count"] += len(artifact_refs)
            stats["max_tool_output_bytes"] = max(
                stats["max_tool_output_bytes"],
                _max_nested_artifact_ref_size_bytes(artifact_refs),
            )
        stats["artifact_spillover_count"] += _count_nested_artifact_refs(output)
        stats["max_tool_output_bytes"] = max(
            stats["max_tool_output_bytes"],
            _max_nested_artifact_ref_size_bytes(output),
        )
        command = _extract_command(entry)
        if command:
            command_counts[command] = command_counts.get(command, 0) + 1
        failure_code = str(entry.get("failure_code") or "")
        if failure_code.startswith("client_execution.tool_protocol"):
            stats["tool_protocol_error_count"] += 1
        if failure_code == "client_execution.tool_argument_invalid":
            stats["parse_error_count"] += 1
    stats["repeated_command_count"] = max(command_counts.values(), default=0)
    return stats


def _count_nested_artifact_refs(payload: Any) -> int:
    if isinstance(payload, Mapping):
        count = 0
        for key, value in payload.items():
            if key in {"artifact_refs", "output_artifact_refs"} and isinstance(value, list):
                count += len(value)
                continue
            count += _count_nested_artifact_refs(value)
        return count
    if isinstance(payload, list):
        return sum(_count_nested_artifact_refs(item) for item in payload)
    return 0


def _max_nested_artifact_ref_size_bytes(payload: Any) -> int:
    if isinstance(payload, Mapping):
        max_size = 0
        for key, value in payload.items():
            if key in {"artifact_refs", "output_artifact_refs"} and isinstance(value, list):
                max_size = max(max_size, _max_artifact_ref_size_bytes(value))
                continue
            max_size = max(max_size, _max_nested_artifact_ref_size_bytes(value))
        return max_size
    if isinstance(payload, list):
        return max((_max_nested_artifact_ref_size_bytes(item) for item in payload), default=0)
    return 0


def _max_artifact_ref_size_bytes(refs: list[Any]) -> int:
    max_size = 0
    for ref in refs:
        if not isinstance(ref, Mapping):
            continue
        try:
            max_size = max(max_size, int(ref.get("size_bytes") or 0))
        except (TypeError, ValueError):
            continue
    return max_size


def _extract_command(entry: Mapping[str, Any]) -> str:
    input_payload = entry.get("input")
    if isinstance(input_payload, Mapping) and isinstance(input_payload.get("command"), str):
        return str(input_payload["command"])
    output_payload = entry.get("output")
    if isinstance(output_payload, Mapping) and isinstance(output_payload.get("command"), str):
        return str(output_payload["command"])
    return ""


def _materialized_artifact_paths(
    session: Any,
    artifact_paths: Mapping[str, Any],
) -> dict[str, str]:
    sample_root = Path(str(getattr(session, "artifact_layout", {}).get("sample_root") or "."))
    resolved: dict[str, str] = {}
    for key, raw_path in artifact_paths.items():
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path)
        absolute = path if path.is_absolute() else sample_root / path
        if absolute.exists():
            resolved[str(key)] = (
                absolute.relative_to(sample_root).as_posix()
                if absolute.is_relative_to(sample_root)
                else str(absolute)
            )
    return resolved


def _resolve_patch_content(output: Mapping[str, Any]) -> str:
    patch_fields = ("patch_content", "patch", "diff", "answer", "text", "content")
    for field in patch_fields:
        value = output.get(field)
        if isinstance(value, str) and value.strip():
            cleaned = _clean_patch_candidate(value)
            if cleaned:
                return cleaned
    agent_trace = output.get("agent_trace")
    if isinstance(agent_trace, list):
        patch_from_trace = _resolve_patch_from_agent_trace(agent_trace)
        if patch_from_trace:
            return patch_from_trace
    return ""


def _resolve_patch_from_agent_trace(agent_trace: list[Any]) -> str:
    for entry in reversed(agent_trace):
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or entry.get("tool") or "")
        if name not in {"submit_patch_tool", "codex_exec"}:
            continue
        output = entry.get("output")
        patch = _clean_patch_candidate(_extract_text_payload(output))
        if patch:
            return patch
    return ""


def _extract_text_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        for key in ("patch_content", "patch", "diff", "stdout", "text", "content", "answer"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        nested = payload.get("output")
        if nested is not None and nested is not payload:
            return _extract_text_payload(nested)
    return ""


def _decode_payload(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return payload.decode("utf-8", errors="replace")
    if payload is None:
        return ""
    return str(payload)


def _clean_patch_candidate(value: str) -> str:
    cleaned = clean_patch_content(value)
    if _looks_like_patch(cleaned):
        return cleaned
    return ""


def _looks_like_patch(value: str) -> bool:
    stripped = value.strip()
    return "diff --git" in stripped or stripped.startswith("*** Begin Patch") or stripped.startswith("--- ")


def _build_trace_diagnostics(agent_trace: Any) -> dict[str, list[str]]:
    if not isinstance(agent_trace, list):
        return {"missing_commands": [], "recent_errors": []}

    missing_commands: list[str] = []
    recent_errors: list[str] = []
    for entry in agent_trace[-5:]:
        if not isinstance(entry, Mapping):
            continue
        output = entry.get("output")
        stderr = ""
        if isinstance(output, Mapping):
            stderr = _decode_payload(output.get("stderr"))
        elif isinstance(entry.get("stderr"), str):
            stderr = str(entry.get("stderr") or "")
        for line in stderr.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            recent_errors.append(stripped)
            match = re.search(r": ([A-Za-z0-9_.-]+): not found$", stripped)
            if match:
                command = match.group(1)
                if command not in missing_commands:
                    missing_commands.append(command)
    return {
        "missing_commands": missing_commands,
        "recent_errors": recent_errors[-5:],
    }


def _write_json_artifact(session: Any, filename: str, payload: Any) -> str:
    target, relative_path = resolve_sample_artifact_target(session, filename)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return relative_path


def _write_text_artifact(session: Any, filename: str, content: str) -> str:
    target, relative_path = resolve_sample_artifact_target(session, filename)
    target.write_text(content, encoding="utf-8")
    return relative_path
