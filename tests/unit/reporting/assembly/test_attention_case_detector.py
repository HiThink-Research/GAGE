from __future__ import annotations

import pytest

from gage_eval.reporting.assembly.attention_detector import AttentionCaseDetector, _severity


def _candidate(reason_code: str, *, frequency: float) -> dict:
    return {
        "case_id": f"task/sample/{reason_code}",
        "task_id": "task",
        "sample_id": "sample",
        "reason_codes": [reason_code],
        "frequency": frequency,
        "summary": reason_code,
        "evidence_ref_ids": ["evidence://sample/task/sample"],
    }


@pytest.mark.fast
def test_attention_case_detector_scores_registered_reason_codes() -> None:
    cases = AttentionCaseDetector().detect(
        [
            {
                "case_id": "task/sample",
                "task_id": "task",
                "sample_id": "sample",
                "reason_codes": ["scheduler.failed"],
                "summary": "Scheduler failed",
                "evidence_ref_ids": ["evidence://sample/task/sample"],
            }
        ],
        total_samples=1,
    )

    assert cases[0].severity.value == "critical"
    assert cases[0].scoring["priority_score"] >= 0.85


@pytest.mark.fast
def test_attention_case_detector_uses_design_weights() -> None:
    cases = AttentionCaseDetector().detect(
        [_candidate("score.low", frequency=0.25)],
        total_samples=10,
    )

    assert cases[0].scoring["priority_score"] == pytest.approx(
        0.30 * 0.25 + 0.50 * 0.60 + 0.20 * 0.65
    )


@pytest.mark.fast
def test_attention_case_detector_maps_critical_and_info_thresholds() -> None:
    critical = AttentionCaseDetector().detect(
        [_candidate("scheduler.failed", frequency=1.0)],
        total_samples=1,
    )[0]
    low = AttentionCaseDetector().detect(
        [_candidate("expected.failure", frequency=0.0)],
        total_samples=100,
    )[0]

    assert critical.scoring["priority_score"] == pytest.approx(0.925)
    assert critical.severity.value == "critical"
    assert low.scoring["priority_score"] == pytest.approx(0.21)
    assert low.severity.value == "low"
    assert _severity(0.19, reason_codes=[]).value == "info"


@pytest.mark.fast
def test_attention_case_detector_direct_hit_reason_codes_are_at_least_high() -> None:
    case = AttentionCaseDetector().detect(
        [_candidate("timeout", frequency=0.0)],
        total_samples=100,
    )[0]

    assert case.scoring["priority_score"] == pytest.approx(0.43)
    assert case.severity.value == "high"
