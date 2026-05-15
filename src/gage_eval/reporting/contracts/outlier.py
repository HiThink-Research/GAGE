from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Diagnostic = dict[str, Any]


class MetricScope:
    """Validates metric scope ownership rules."""

    VALUES = ("run", "task", "section")

    @classmethod
    def validate(cls, metric: dict[str, Any], path: str = "metrics[]") -> list[Diagnostic]:
        """Returns diagnostics for invalid metric scope declarations."""
        diagnostics: list[Diagnostic] = []
        scope = metric.get("scope")
        if scope not in cls.VALUES:
            diagnostics.append(
                {
                    "code": "report_context.invalid_enum",
                    "path": f"{path}.scope",
                    "message": f"Unsupported metric scope: {scope}",
                }
            )
            return diagnostics
        if scope == "task" and not metric.get("task_id"):
            diagnostics.append(
                {"code": "report_context.required_missing", "path": f"{path}.task_id"}
            )
        if scope == "section" and not metric.get("section_id"):
            diagnostics.append(
                {"code": "report_context.required_missing", "path": f"{path}.section_id"}
            )
        return diagnostics


@dataclass
class OutlierEntry:
    """Represents one item in an outlier top-k list."""

    sample_id: str | None = None
    value: float | int | None = None
    trial_id: str | None = None
    p_rank: float | None = None
    evidence_ref_ids: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OutlierEntry":
        """Builds an outlier entry from a mapping."""
        payload = dict(data or {})
        known = {field_name: payload.pop(field_name, None) for field_name in _ENTRY_FIELDS}
        known["evidence_ref_ids"] = list(known["evidence_ref_ids"] or [])
        return cls(extra=payload, **known)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the outlier entry."""
        data = {field_name: getattr(self, field_name) for field_name in _ENTRY_FIELDS}
        data["evidence_ref_ids"] = list(self.evidence_ref_ids)
        data = {key: value for key, value in data.items() if value is not None and value != []}
        data.update(self.extra)
        return data

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass
class OutlierGroup:
    """Groups top-k outliers for a metric."""

    metric_id: str | None = None
    scope: str | None = None
    ranking: str | None = None
    top_k: list[OutlierEntry] = field(default_factory=list)
    task_id: str | None = None
    section_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OutlierGroup":
        """Builds an outlier group from a mapping."""
        payload = dict(data or {})
        top_k = [OutlierEntry.from_dict(item) for item in payload.pop("top_k", [])]
        known = {field_name: payload.pop(field_name, None) for field_name in _GROUP_FIELDS}
        known["top_k"] = top_k
        return cls(extra=payload, **known)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the outlier group."""
        data = {field_name: getattr(self, field_name) for field_name in _GROUP_FIELDS}
        data["top_k"] = [item.to_dict() for item in self.top_k]
        data = {key: value for key, value in data.items() if value is not None}
        data.update(self.extra)
        return data

    def validate(self, path: str = "outliers[]") -> list[Diagnostic]:
        """Returns diagnostics for invalid outlier groups."""
        diagnostics: list[Diagnostic] = []
        for field_name in ("metric_id", "scope", "ranking", "top_k"):
            value = getattr(self, field_name)
            if value is None or value == []:
                diagnostics.append(
                    {
                        "code": "report_context.required_missing",
                        "path": f"{path}.{field_name}",
                    }
                )
        diagnostics.extend(MetricScope.validate(self.to_dict(), path))
        return diagnostics

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


_ENTRY_FIELDS = ("sample_id", "trial_id", "value", "p_rank", "evidence_ref_ids")
_GROUP_FIELDS = ("metric_id", "scope", "task_id", "section_id", "ranking")
