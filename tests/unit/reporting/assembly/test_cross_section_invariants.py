from __future__ import annotations

import pytest

from gage_eval.reporting.contracts import ReportContext


@pytest.mark.fast
def test_cross_section_validator_reports_missing_evidence_refs_and_secret() -> None:
    payload = ReportContext.minimal("run").to_dict()
    payload["attention_cases"] = [
        {
            "case_id": "case",
            "severity": "high",
            "scoring": {"frequency": 1, "impact": "high", "actionability": "high", "priority_score": 0.9},
            "reason_codes": ["scheduler.failed"],
            "summary": "Bearer abc123",
            "evidence_ref_ids": ["evidence://missing"],
        }
    ]

    errors = ReportContext.validate(payload)

    assert any("evidence" in error for error in errors)
    assert any("secret" in error for error in errors)
