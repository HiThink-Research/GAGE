from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gage_eval.reporting.contracts.severity import Severity


@dataclass
class FailureCluster:
    """Represents a grouped failure pattern derived from reason codes."""

    cluster_id: str | None = None
    cluster_key: list[str] = field(default_factory=list)
    count: int | None = None
    severity: str | None = None
    sample_ids: list[str] = field(default_factory=list)
    representative_ref_ids: list[str] = field(default_factory=list)
    label: str | None = None
    hypothesis: str | None = None
    recommended_action: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FailureCluster":
        """Builds a failure cluster from a mapping."""
        payload = dict(data or {})
        known = {field_name: payload.pop(field_name, None) for field_name in _FIELDS}
        for field_name in ("cluster_key", "sample_ids", "representative_ref_ids"):
            known[field_name] = list(known[field_name] or [])
        return cls(extra=payload, **known)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the failure cluster."""
        data = {field_name: getattr(self, field_name) for field_name in _FIELDS}
        data = {key: value for key, value in data.items() if value is not None}
        data.update(self.extra)
        return data

    def validate(self, path: str = "failure_clusters[]") -> list[dict[str, Any]]:
        """Returns diagnostics for invalid failure clusters."""
        diagnostics: list[dict[str, Any]] = []
        for field_name in (
            "cluster_id",
            "cluster_key",
            "count",
            "severity",
            "sample_ids",
            "representative_ref_ids",
        ):
            value = getattr(self, field_name)
            if value is None or value == []:
                diagnostics.append(
                    {
                        "code": "report_context.required_missing",
                        "path": f"{path}.{field_name}",
                    }
                )
        if self.severity is not None:
            diagnostics.extend(Severity.validate(self.severity, f"{path}.severity"))
        return diagnostics


_FIELDS = (
    "cluster_id",
    "cluster_key",
    "label",
    "count",
    "severity",
    "sample_ids",
    "representative_ref_ids",
    "hypothesis",
    "recommended_action",
)
