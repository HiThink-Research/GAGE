from __future__ import annotations

from typing import Any, Mapping


TOOL_PROTOCOL_FAILURE_CODES = {
    "client_execution.tool_protocol_parse_error",
    "client_execution.tool_protocol_missing_call",
    "client_execution.tool_protocol_missing_call_id",
    "client_execution.tool_router.not_found",
    "client_execution.tool_result_injection_failed",
    "client_execution.tool_retry_budget_exhausted",
}


def resolve_swebench_failure_category(payload: Mapping[str, Any] | None) -> str:
    data = dict(payload or {})
    explicit = data.get("failure_category") or data.get("final_failure_category")
    if explicit:
        return str(explicit)

    failure_code = str(data.get("failure_code") or "")
    failure_reason = str(data.get("failure_reason") or data.get("diagnostic_reason") or "")

    if failure_code == "artifact_capture.patch_missing" or failure_reason == "missing_patch":
        return "missing_patch"
    if failure_reason in {"missing_metadata", "missing_base_commit"}:
        return "missing_metadata"
    if failure_reason == "missing_run_scripts":
        return "missing_run_scripts"
    if failure_reason == "missing_output":
        return "missing_output"
    if failure_reason == "invalid_output":
        return "invalid_output"
    if failure_reason == "assertion_error":
        return "wrong_solution"
    if failure_code.startswith("environment.") or failure_reason in {"sandbox_judge_error", "container_error"}:
        return "sandbox_judge_error"
    if _looks_like_context_overflow_from_listing(data):
        return "context_overflow_from_listing"
    if _looks_like_endless_file_reading(data):
        return "endless_file_reading"
    if _looks_like_syntax_error(data):
        return "syntax_error"
    if failure_code in TOOL_PROTOCOL_FAILURE_CODES:
        return "tool_protocol_error"
    if failure_code == "client_execution.tool_argument_invalid" or failure_reason == "parse_error":
        return "parse_error"
    if failure_reason == "test_execution_error" or failure_code.startswith("verifier.executor."):
        return "test_execution_error"
    return "unknown"


def _looks_like_context_overflow_from_listing(data: Mapping[str, Any]) -> bool:
    if not _looks_like_failed_run(data):
        return False
    spillover_count = _int_value(data.get("artifact_spillover_count"))
    max_tool_output_bytes = _int_value(data.get("max_tool_output_bytes"))
    reason = str(data.get("failure_reason") or data.get("diagnostic_reason") or "").lower()
    if "context" in reason and "overflow" in reason:
        return True
    return spillover_count >= 3 or max_tool_output_bytes >= 100_000


def _looks_like_endless_file_reading(data: Mapping[str, Any]) -> bool:
    if not _looks_like_failed_run(data):
        return False
    repeated_command_count = max(
        _int_value(data.get("repeated_command_count")),
        _int_value(data.get("same_command_repetition_count")),
        _int_value(data.get("max_repeated_command_count")),
    )
    loop_exit_reason = str(data.get("loop_exit_reason") or "").lower()
    return repeated_command_count >= 8 and loop_exit_reason in {"max_turns", "tool_call_retry_budget", "turn_limit"}


def _looks_like_syntax_error(data: Mapping[str, Any]) -> bool:
    if not _looks_like_failed_run(data):
        return False
    if _int_value(data.get("parse_error_count")) < 1:
        return False
    text = " ".join(str(item) for item in data.get("recent_errors") or [])
    text += " " + str(data.get("failure_reason") or "")
    return "syntaxerror" in text.lower() or "syntax error" in text.lower()


def _looks_like_failed_run(data: Mapping[str, Any]) -> bool:
    status = str(data.get("status") or "").lower()
    if data.get("resolved") is True or status in {"completed", "passed", "success"}:
        return False
    return bool(
        data.get("failure_code")
        or data.get("failure_reason")
        or data.get("diagnostic_reason")
        or data.get("loop_exit_reason")
    )


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
