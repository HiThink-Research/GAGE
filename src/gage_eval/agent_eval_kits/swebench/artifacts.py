from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from gage_eval.agent_eval_kits.common import resolve_sample_artifact_target

_TOOL_ARGUMENT_INVALID_FAILURE_CATEGORY = "client_execution.tool_argument_invalid"
_TOOL_PROTOCOL_PARSE_ERROR_FAILURE_CATEGORY = "client_execution.tool_protocol_parse_error"
_TOOL_PROTOCOL_MISSING_CALL_FAILURE_CATEGORY = "client_execution.tool_protocol_missing_call"
_PATCH_MISSING_FAILURE_CATEGORY = "artifact_capture.patch_missing"
_STRUCTURED_INVALID_TOOL_ERRORS = {
    "command_missing",
    "path_missing",
    "replace_in_file_missing_args",
    "meta_tool_payload_invalid",
    "meta_tool_endpoint_missing",
    "meta_tool_params_invalid",
}
_STRUCTURED_INVALID_TOOL_ERROR_PREFIXES = ("invalid_endpoint:",)


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
        sandbox_provider: Optional sandbox provider for patch extraction.

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
    patch_content = _resolve_patch_content(output, sandbox_provider=sandbox_provider)
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
    resolved_failure_category = resolve_swebench_failure_category(
        output=output,
        agent_trace=agent_trace,
        materialized_artifact_paths=persisted,
    )
    if resolved_failure_category:
        output.setdefault("failure_category", resolved_failure_category)
    diagnostics = _build_swebench_diagnostics(
        session=session,
        output=output,
        agent_trace=agent_trace,
        materialized_artifact_paths=persisted,
        trace_diagnostics=trace_diagnostics,
        answer=answer,
    )
    persisted["swebench_diagnostics"] = _write_json_artifact(
        session,
        "swebench_diagnostics.json",
        diagnostics,
    )

    return persisted


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


def _resolve_patch_content(
    output: Mapping[str, Any],
    *,
    sandbox_provider: Any,
) -> str:
    patch_fields = ("patch_content", "patch", "diff")
    for field in patch_fields:
        value = output.get(field)
        if isinstance(value, str) and value.strip():
            return _normalize_text_block(value)
    agent_trace = output.get("agent_trace")
    if isinstance(agent_trace, list):
        patch_from_trace = _resolve_patch_from_agent_trace(agent_trace)
        if patch_from_trace:
            return patch_from_trace
    sandbox_patch = _read_submission_patch_from_sandbox(sandbox_provider)
    if sandbox_patch:
        return sandbox_patch
    return _read_git_diff_from_sandbox(sandbox_provider)


def _resolve_patch_from_agent_trace(agent_trace: list[Any]) -> str:
    for entry in reversed(agent_trace):
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or entry.get("tool") or "")
        if name not in {"submit_patch_tool", "codex_exec"}:
            continue
        output = entry.get("output")
        patch = _extract_text_payload(output)
        if "diff --git" in patch or patch.startswith("*** Begin Patch"):
            return _normalize_text_block(patch)
    return ""


def _read_submission_patch_from_sandbox(sandbox_provider: Any) -> str:
    if sandbox_provider is None:
        return ""
    handle = getattr(sandbox_provider, "get_handle", lambda: None)()
    sandbox = getattr(handle, "sandbox", None) if handle is not None else None
    if sandbox is None:
        return ""
    reader = getattr(sandbox, "read_file", None)
    if callable(reader):
        try:
            payload = reader("/workspace/submission.patch")
        except Exception:
            payload = ""
        return _normalize_text_block(_decode_payload(payload))
    executor = getattr(sandbox, "exec", None)
    if not callable(executor):
        return ""
    try:
        result = executor("cat /workspace/submission.patch", timeout=5)
    except Exception:
        return ""
    stdout = getattr(result, "stdout", "")
    return _normalize_text_block(_decode_payload(stdout))


def _read_git_diff_from_sandbox(sandbox_provider: Any) -> str:
    if sandbox_provider is None:
        return ""
    handle = getattr(sandbox_provider, "get_handle", lambda: None)()
    sandbox = getattr(handle, "sandbox", None) if handle is not None else None
    executor = getattr(sandbox, "exec", None)
    if not callable(executor):
        return ""
    commands = (
        "cd /workspace && git diff --binary --no-color",
        "cd /workspace && git diff --binary --no-color HEAD --",
    )
    for command in commands:
        try:
            result = executor(command, timeout=10)
        except Exception:
            continue
        stdout = _normalize_text_block(_decode_payload(getattr(result, "stdout", "")))
        if "diff --git" in stdout or stdout.startswith("*** Begin Patch"):
            return stdout
    return ""


def _extract_text_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        for key in ("patch", "diff", "stdout", "text", "content", "answer"):
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


def _normalize_text_block(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if not text.endswith("\n"):
        text += "\n"
    return text


def _build_trace_diagnostics(agent_trace: Any) -> dict[str, list[str]]:
    if not isinstance(agent_trace, list):
        return {"missing_commands": [], "recent_errors": []}

    missing_commands: list[str] = []
    recent_errors: list[str] = []
    for entry in agent_trace:
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
            match = re.search(r": ([A-Za-z0-9_.-]+): not found$", stripped)
            if match:
                command = match.group(1)
                if command not in missing_commands:
                    missing_commands.append(command)
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
            if stripped:
                recent_errors.append(stripped)
    return {
        "missing_commands": missing_commands,
        "recent_errors": recent_errors[-5:],
    }


def _build_swebench_diagnostics(
    *,
    session: Any,
    output: Mapping[str, Any],
    agent_trace: Any,
    materialized_artifact_paths: Mapping[str, str],
    trace_diagnostics: Mapping[str, list[str]],
    answer: Any,
) -> dict[str, Any]:
    trace_stats = _collect_trace_stats(agent_trace)
    prompt_present, prompt_source = _resolve_prompt_metadata(session, output)
    return {
        "prompt_present": prompt_present,
        "prompt_source": prompt_source,
        "input_failure_code": output.get("input_failure_code"),
        "tool_call_retry_count": trace_stats["tool_call_retry_count"],
        "tool_call_retry_total": trace_stats["tool_call_retry_total"],
        "tool_call_parse_error_count": trace_stats["tool_call_parse_error_count"],
        "largest_tool_output_bytes": trace_stats["largest_tool_output_bytes"],
        "largest_tool_name": trace_stats["largest_tool_name"],
        "largest_command_preview": trace_stats["largest_command_preview"],
        "artifact_spillovers": _count_artifact_spillovers(materialized_artifact_paths),
        "final_failure_category": resolve_swebench_failure_category(
            output=output,
            agent_trace=agent_trace,
            materialized_artifact_paths=materialized_artifact_paths,
        ),
        "submission_patch_present": "submission_patch" in materialized_artifact_paths,
        "agent_trace_step_count": len(agent_trace) if isinstance(agent_trace, list) else 0,
        "answer_present": isinstance(answer, str) and bool(answer.strip()),
        "required_artifact": "submission.patch",
        "missing_commands": trace_diagnostics["missing_commands"],
        "recent_errors": trace_diagnostics["recent_errors"],
    }


def _resolve_prompt_metadata(session: Any, output: Mapping[str, Any]) -> tuple[Any, Any]:
    prompt_present = output.get("prompt_present")
    prompt_source = output.get("prompt_source")
    if prompt_present is not None and prompt_source is not None:
        return prompt_present, prompt_source
    for attr in ("prompt_context", "runtime_context"):
        context = getattr(session, attr, None)
        if not isinstance(context, Mapping):
            continue
        if prompt_present is None:
            prompt_present = context.get("prompt_present")
        if prompt_source is None:
            prompt_source = context.get("prompt_source")
        if prompt_present is not None and prompt_source is not None:
            break
    return prompt_present, prompt_source


def _collect_trace_stats(agent_trace: Any) -> dict[str, Any]:
    stats = {
        "tool_call_retry_count": 0,
        "tool_call_retry_total": 0,
        "tool_call_parse_error_count": 0,
        "largest_tool_output_bytes": 0,
        "largest_tool_name": "",
        "largest_command_preview": "",
        "truncated_tool_count": 0,
    }
    if not isinstance(agent_trace, list):
        return stats

    for entry in agent_trace:
        if not isinstance(entry, Mapping):
            continue
        output = entry.get("output")
        if isinstance(output, Mapping):
            # AgentLoop writes these as monotonic counters, so the latest/max value
            # already represents the total retries seen in the trace.
            retry_count = output.get("retry_count")
            if isinstance(retry_count, int) and retry_count > stats["tool_call_retry_count"]:
                stats["tool_call_retry_count"] = retry_count
            retry_total = output.get("total_invalid_tool_calls")
            if isinstance(retry_total, int) and retry_total > stats["tool_call_retry_total"]:
                stats["tool_call_retry_total"] = retry_total
            if output.get("tool_call_parse_error_type"):
                stats["tool_call_parse_error_count"] += 1
        if entry.get("trace_role") != "tool" or not isinstance(output, Mapping):
            continue
        if output.get("truncated") is True:
            stats["truncated_tool_count"] += 1
        output_bytes = _estimate_tool_output_bytes(output)
        if output_bytes < stats["largest_tool_output_bytes"]:
            continue
        stats["largest_tool_output_bytes"] = output_bytes
        stats["largest_tool_name"] = str(entry.get("name") or entry.get("tool") or "")
        stats["largest_command_preview"] = _extract_command_preview(entry.get("input"))
    return stats


def _estimate_tool_output_bytes(output: Mapping[str, Any]) -> int:
    total = 0
    has_stdio = False
    for key in ("stdout", "stderr"):
        original_length = output.get(f"{key}_original_length")
        if isinstance(original_length, int) and original_length >= 0:
            total += original_length
            has_stdio = True
            continue
        value = output.get(key)
        if isinstance(value, str):
            total += len(value.encode("utf-8"))
            has_stdio = True
    if has_stdio:
        return total
    return len(json.dumps(output, ensure_ascii=False, default=str).encode("utf-8"))


def resolve_swebench_failure_category(
    *,
    output: Mapping[str, Any],
    agent_trace: Any,
    materialized_artifact_paths: Mapping[str, str],
) -> str | None:
    runtime_failure = output.get("runtime_failure")
    if isinstance(runtime_failure, Mapping):
        failure_code = runtime_failure.get("failure_code")
        if isinstance(failure_code, str) and failure_code.strip():
            return failure_code

    failure_category = output.get("failure_category")
    if isinstance(failure_category, str) and failure_category.strip():
        return failure_category

    input_failure_code = output.get("input_failure_code")
    if isinstance(input_failure_code, str) and input_failure_code.strip():
        return input_failure_code

    if "submission_patch" in materialized_artifact_paths:
        return None

    trace_failure_category = _resolve_trace_failure_category(agent_trace)
    if trace_failure_category:
        return trace_failure_category

    if "submission_patch" not in materialized_artifact_paths:
        return _PATCH_MISSING_FAILURE_CATEGORY
    return None


def _resolve_trace_failure_category(agent_trace: Any) -> str | None:
    if not isinstance(agent_trace, list):
        return None

    saw_missing_tool_call = False
    for entry in agent_trace:
        if not isinstance(entry, Mapping):
            continue
        output = entry.get("output")
        if not isinstance(output, Mapping):
            continue
        if _is_tool_argument_invalid_output(output):
            return _TOOL_ARGUMENT_INVALID_FAILURE_CATEGORY
        if output.get("tool_call_parse_error_type"):
            return _TOOL_PROTOCOL_PARSE_ERROR_FAILURE_CATEGORY
        invalid_tool_names = output.get("invalid_tool_call_names")
        if isinstance(invalid_tool_names, list) and invalid_tool_names:
            saw_missing_tool_call = True
        if entry.get("status") == "retry_required_tool_call":
            saw_missing_tool_call = True
    if saw_missing_tool_call:
        return _TOOL_PROTOCOL_MISSING_CALL_FAILURE_CATEGORY
    return None


def _is_tool_argument_invalid_output(output: Mapping[str, Any]) -> bool:
    error_code = output.get("error_code")
    if isinstance(error_code, str) and error_code == "tool_argument_invalid":
        return True
    error = output.get("error")
    if not isinstance(error, str):
        return False
    if error in _STRUCTURED_INVALID_TOOL_ERRORS:
        return True
    return any(error.startswith(prefix) for prefix in _STRUCTURED_INVALID_TOOL_ERROR_PREFIXES)


def _extract_command_preview(payload: Any, limit: int = 160) -> str:
    command = ""
    if isinstance(payload, Mapping):
        for key in ("command", "cmd"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                command = value.strip()
                break
    if not command:
        return ""
    if len(command) <= limit:
        return command
    return f"{command[: limit - 3]}..."


def _count_artifact_spillovers(materialized_artifact_paths: Mapping[str, str]) -> int:
    reserved = {"submission_patch", "agent_trace", "final_response", "swebench_diagnostics"}
    return sum(1 for key in materialized_artifact_paths if key not in reserved)


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
