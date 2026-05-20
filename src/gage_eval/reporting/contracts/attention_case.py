from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gage_eval.reporting.contracts.severity import Severity


Diagnostic = dict[str, Any]


@dataclass
class AttentionCaseScoring:
    """Stores standardized attention-case scoring factors."""

    frequency: float | None = None
    impact: str | None = None
    actionability: str | None = None
    priority_score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AttentionCaseScoring":
        """Builds scoring factors from a JSON-compatible mapping."""
        payload = dict(data or {})
        known = {field_name: payload.pop(field_name, None) for field_name in _SCORING_FIELDS}
        return cls(extra=payload, **known)

    def to_dict(self) -> dict[str, Any]:
        """Serializes scoring factors."""
        data = {field_name: getattr(self, field_name) for field_name in _SCORING_FIELDS}
        data.update(self.extra)
        return data

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def validate(self, path: str = "attention_cases[].scoring") -> list[Diagnostic]:
        """Returns diagnostics for missing scoring factors."""
        diagnostics: list[Diagnostic] = []
        for field_name in _SCORING_FIELDS:
            if getattr(self, field_name) is None:
                diagnostics.append(
                    {
                        "code": "report_context.required_missing",
                        "path": f"{path}.{field_name}",
                    }
                )
        return diagnostics


@dataclass
class AttentionCase:
    """Represents a prioritized case for human review."""

    case_id: str | None = None
    severity: str | None = None
    scoring: AttentionCaseScoring | None = None
    reason_codes: list[str] = field(default_factory=list)
    summary: str | None = None
    evidence_ref_ids: list[str] = field(default_factory=list)
    task_id: str | None = None
    sample_id: str | None = None
    trial_id: str | None = None
    first_seen_in_sample: str | None = None
    expected_failure: bool = False
    excluded_reason_codes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AttentionCase":
        """Builds an attention case from a JSON-compatible mapping."""
        payload = dict(data or {})
        scoring = AttentionCaseScoring.from_dict(payload.pop("scoring", None))
        known = {field_name: payload.pop(field_name, None) for field_name in _ATTENTION_FIELDS}
        known["scoring"] = scoring
        known["reason_codes"] = list(known["reason_codes"] or [])
        known["evidence_ref_ids"] = list(known["evidence_ref_ids"] or [])
        known["excluded_reason_codes"] = list(known["excluded_reason_codes"] or [])
        if known["expected_failure"] is None:
            known["expected_failure"] = False
        return cls(extra=payload, **known)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the attention case."""
        severity = self.severity.value if hasattr(self.severity, "value") else self.severity
        data: dict[str, Any] = {
            "case_id": self.case_id,
            "severity": severity,
            "scoring": self.scoring.to_dict() if self.scoring else None,
            "reason_codes": list(self.reason_codes),
            "summary": self.summary,
            "evidence_ref_ids": list(self.evidence_ref_ids),
            "expected_failure": self.expected_failure,
            "excluded_reason_codes": list(self.excluded_reason_codes),
        }
        for field_name in ("task_id", "sample_id", "trial_id", "first_seen_in_sample"):
            value = getattr(self, field_name)
            if value is not None:
                data[field_name] = value
        data.update(self.extra)
        return data

    def validate(self, path: str = "attention_cases[]") -> list[Diagnostic]:
        """Returns diagnostics for invalid attention cases."""
        diagnostics: list[Diagnostic] = []
        for field_name in ("case_id", "severity", "scoring", "reason_codes", "summary", "evidence_ref_ids"):
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
        if self.scoring is not None:
            diagnostics.extend(self.scoring.validate(f"{path}.scoring"))
        return diagnostics

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


_SCORING_FIELDS = ("frequency", "impact", "actionability", "priority_score")

_ATTENTION_FIELDS = (
    "case_id",
    "task_id",
    "sample_id",
    "trial_id",
    "first_seen_in_sample",
    "severity",
    "reason_codes",
    "expected_failure",
    "excluded_reason_codes",
    "summary",
    "evidence_ref_ids",
)
