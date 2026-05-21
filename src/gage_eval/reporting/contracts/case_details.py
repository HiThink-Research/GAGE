from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaseDetails:
    """Stores bounded drill-down previews for one attention case."""

    message_history_preview: list[dict[str, Any]] = field(default_factory=list)
    tool_call_summary: list[dict[str, Any]] = field(default_factory=list)
    scoring_breakdown: dict[str, Any] = field(default_factory=dict)
    artifact_previews: list[dict[str, Any]] = field(default_factory=list)
    artifact_preview_ref_ids: list[str] = field(default_factory=list)
    evidence_ref_ids: list[str] = field(default_factory=list)
    truncated: bool = False
    full_trace_ref_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CaseDetails":
        """Builds case details from a JSON-compatible mapping."""
        payload = dict(data or {})
        known = {field_name: payload.pop(field_name, None) for field_name in _FIELDS}
        for field_name in _LIST_FIELDS:
            known[field_name] = list(known[field_name] or [])
        known["scoring_breakdown"] = dict(known["scoring_breakdown"] or {})
        known["truncated"] = bool(known["truncated"])
        return cls(extra=payload, **known)

    def to_dict(self) -> dict[str, Any]:
        """Serializes case details."""
        data = {field_name: getattr(self, field_name) for field_name in _FIELDS}
        data = {key: value for key, value in data.items() if value is not None}
        data.update(self.extra)
        return data

    def validate(self, path: str = "case_details{}") -> list[dict[str, Any]]:
        """Returns diagnostics for invalid case details."""
        if self.truncated is None:
            return [{"code": "report_context.required_missing", "path": f"{path}.truncated"}]
        return []


_LIST_FIELDS = (
    "message_history_preview",
    "tool_call_summary",
    "artifact_previews",
    "artifact_preview_ref_ids",
    "evidence_ref_ids",
)
_FIELDS = (
    *_LIST_FIELDS,
    "scoring_breakdown",
    "truncated",
    "full_trace_ref_id",
)
