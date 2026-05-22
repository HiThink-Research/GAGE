from __future__ import annotations

from typing import Any

from gage_eval.reporting.contracts import CaseDetails
from gage_eval.reporting.privacy import SecretFilter


_SECRET_FILTER = SecretFilter()


class CaseDetailsBuilder:
    def __init__(self, max_messages: int = 20, max_tool_calls: int = 20, max_preview_bytes: int = 4096) -> None:
        self.max_messages = max_messages
        self.max_tool_calls = max_tool_calls
        self.max_preview_bytes = max_preview_bytes

    def build(self, record: dict[str, Any]) -> CaseDetails:
        messages = list(record.get("messages") or [])[: self.max_messages]
        tool_calls = list(record.get("tool_calls") or [])[: self.max_tool_calls]
        details = CaseDetails(
            message_history_preview=_redact(_truncate(messages, self.max_preview_bytes)),
            tool_call_summary=_redact(_truncate(tool_calls, self.max_preview_bytes)),
            scoring_breakdown=_redact(record.get("scoring_breakdown") or {}),
            artifact_preview_ref_ids=list(record.get("artifact_preview_ref_ids") or []),
            evidence_ref_ids=list(record.get("evidence_ref_ids") or []),
            full_trace_ref_id=record.get("full_trace_ref_id"),
            truncated=len(record.get("messages") or []) > self.max_messages
            or len(record.get("tool_calls") or []) > self.max_tool_calls
            or len(str(record)) > self.max_preview_bytes,
        )
        return details


def _redact(value: Any) -> Any:
    return _SECRET_FILTER.redact(value).value


def _truncate(value: Any, max_bytes: int) -> Any:
    if isinstance(value, list):
        return [_truncate(item, max_bytes) for item in value]
    if isinstance(value, dict):
        return {key: _truncate(item, max_bytes) for key, item in value.items()}
    if not isinstance(value, str):
        return value
    if len(value.encode("utf-8")) <= max_bytes:
        return value
    return value[:max_bytes]
