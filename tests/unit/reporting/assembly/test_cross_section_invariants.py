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


@pytest.mark.fast
def test_cross_section_validator_reports_missing_scenario_profile_refs() -> None:
    payload = ReportContext.minimal("run").to_dict()
    payload["evidence_refs"] = [
        {
            "ref_id": "evidence://artifact/present",
            "kind": "artifact",
            "path": "artifacts/present.json",
            "mime_type": "application/json",
            "sha256": "0" * 64,
            "size_bytes": 2,
            "timestamp_iso": "2026-05-15T00:00:00Z",
        }
    ]
    payload["scenario_profiles"] = {
        "agent": {
            "representative_ref_ids": [
                "evidence://artifact/present",
                "evidence://artifact/missing-agent",
            ],
            "trace_label": "evidence://artifact/not-a-ref-field",
        },
        "game": {
            "replay_refs": [
                "evidence://artifact/missing-replay",
                "runs/demo/replay.json",
            ],
        },
        "media": {
            "example_ref_id": "evidence://media/missing-media",
            "thumbnail_url": "https://example.test/image.png",
        },
    }

    errors = ReportContext.validate(payload)

    assert any(
        "scenario_profiles.agent.representative_ref_ids[1]" in error
        and "evidence://artifact/missing-agent" in error
        for error in errors
    )
    assert any(
        "scenario_profiles.game.replay_refs[0]" in error
        and "evidence://artifact/missing-replay" in error
        for error in errors
    )
    assert any(
        "scenario_profiles.media.example_ref_id" in error
        and "evidence://media/missing-media" in error
        for error in errors
    )
    assert not any("not-a-ref-field" in error for error in errors)
    assert not any("thumbnail_url" in error for error in errors)


@pytest.mark.fast
def test_cross_section_validator_does_not_treat_task_ids_as_secret_markers() -> None:
    payload = ReportContext.minimal("run").to_dict()
    payload["attention_cases"] = [
        {
            "case_id": "task-1/sample-1",
            "task_id": "task-1",
            "sample_id": "sample-1",
            "severity": "high",
            "scoring": {"frequency": 1, "impact": "high", "actionability": "high", "priority_score": 0.9},
            "reason_codes": ["scheduler.failed"],
            "summary": "Sample failed.",
            "evidence_ref_ids": [],
        }
    ]

    errors = ReportContext.validate(payload)

    assert not any("secret" in error for error in errors)
