from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gage_eval.reporting.contracts.attention_case import AttentionCase
from gage_eval.reporting.contracts.case_details import CaseDetails
from gage_eval.reporting.contracts.failure_cluster import FailureCluster
from gage_eval.reporting.contracts.outlier import OutlierGroup


@dataclass
class SummaryGeneratorResult:
    """Represents a summary generator's structured report contribution."""

    generator_id: str | None = None
    summary_sections: list[dict[str, Any]] = field(default_factory=list)
    attention_cases: list[AttentionCase] = field(default_factory=list)
    outliers: list[OutlierGroup] = field(default_factory=list)
    case_details: dict[str, CaseDetails] = field(default_factory=dict)
    reason_code_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    failure_clusters: list[FailureCluster] = field(default_factory=list)
    evidence_ref_ids: list[str] = field(default_factory=list)
    legacy_payload: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SummaryGeneratorResult":
        """Builds a summary result from a JSON-compatible mapping."""
        payload = dict(data or {})
        return cls(
            generator_id=payload.pop("generator_id", None),
            summary_sections=list(payload.pop("summary_sections", [])),
            attention_cases=[
                AttentionCase.from_dict(item)
                for item in payload.pop("attention_cases", [])
            ],
            outliers=[OutlierGroup.from_dict(item) for item in payload.pop("outliers", [])],
            case_details={
                key: CaseDetails.from_dict(value)
                for key, value in payload.pop("case_details", {}).items()
            },
            reason_code_counts=dict(payload.pop("reason_code_counts", {})),
            failure_clusters=[
                FailureCluster.from_dict(item)
                for item in payload.pop("failure_clusters", [])
            ],
            evidence_ref_ids=list(payload.pop("evidence_ref_ids", [])),
            legacy_payload=dict(payload.pop("legacy_payload", {})),
            diagnostics=dict(payload.pop("diagnostics", {})),
            extra=payload,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the summary result."""
        data: dict[str, Any] = {
            "generator_id": self.generator_id,
            "summary_sections": list(self.summary_sections),
            "attention_cases": [_to_dict(item) for item in self.attention_cases],
            "outliers": [_to_dict(item) for item in self.outliers],
            "case_details": {
                key: _to_dict(value) for key, value in self.case_details.items()
            },
            "reason_code_counts": dict(self.reason_code_counts),
            "failure_clusters": [_to_dict(item) for item in self.failure_clusters],
            "evidence_ref_ids": list(self.evidence_ref_ids),
            "legacy_payload": dict(self.legacy_payload),
            "diagnostics": dict(self.diagnostics),
        }
        data.update(self.extra)
        return data


def _to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(value)
