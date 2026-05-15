from __future__ import annotations

import json

import pytest

from gage_eval.reporting.assembly.context_builder import ReportContextBuilder
from gage_eval.reporting.contracts import EvidenceRef, SummaryGeneratorResult
from gage_eval.reporting.evidence.reader import ReportEvidenceReader


@pytest.mark.io
def test_context_builder_outputs_minimal_report_context(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"sample_count": 1}), encoding="utf-8")
    (run_dir / "samples.jsonl").write_text(json.dumps({"sample_id": "sample-1", "status": "completed"}) + "\n", encoding="utf-8")
    index = ReportEvidenceReader().build_index(run_dir)

    context = ReportContextBuilder().build(
        index=index,
        summary_payload={"sample_count": 1, "metrics": []},
        metrics=[],
        tasks=[],
        runtime_health={"sample_count": 1, "completed_count": 1, "failed_count": 0, "aborted_count": 0},
        observability_health={"events_emitted_total": 0},
        generator_result=None,
    )

    payload = context.to_dict()
    assert payload["headline"]["verdict"] == "passed"
    assert payload["scoring_config"]["weights"]["impact"] == 0.5
    assert payload["scoring_config"] == {
        "formula": "0.30*frequency + 0.50*impact_weight + 0.20*actionability_weight",
        "weights": {"frequency": 0.3, "impact": 0.5, "actionability": 0.2},
        "impact_weights": {
            "critical": 1.0,
            "high": 0.85,
            "medium": 0.6,
            "low": 0.3,
            "unknown": 0.4,
        },
        "actionability_weights": {
            "high": 1.0,
            "medium": 0.65,
            "low": 0.3,
            "unknown": 0.4,
        },
    }
    assert payload["methodology"] is not None


@pytest.mark.fast
def test_context_builder_normalizes_task_contract_fields(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "samples.jsonl").write_text(
        json.dumps({"sample_id": "sample-1", "model_output": {"answer": "A"}}) + "\n",
        encoding="utf-8",
    )
    index = ReportEvidenceReader().build_index(run_dir)

    context = ReportContextBuilder().build(
        index=index,
        summary_payload={"sample_count": 1, "metrics": []},
        metrics=[{"metric_id": "acc", "values": {"acc": "1.00000"}}],
        tasks=[
            {
                "task_id": "task-1",
                "metrics": [{"metric_id": "acc", "values": {"acc": "1.00000"}}],
                "execution": {"status": "completed"},
                "sample_count": 1,
            }
        ],
        runtime_health={"sample_count": 1, "completed_count": 1, "failed_count": 0, "aborted_count": 0},
        observability_health={"events_emitted_total": 0},
        generator_result=None,
    )

    payload = context.to_dict()
    assert payload["metrics"][0]["scope"] == "run"
    assert payload["tasks"][0]["status"] == "completed"
    assert payload["tasks"][0]["runtime_health"]["completed_count"] == 1
    assert payload["tasks"][0]["attention_case_count"] == 0
    assert payload["tasks"][0]["failure_cluster_count"] == 0
    assert payload["tasks"][0]["metrics"][0]["scope"] == "task"
    assert payload["diagnostics"]["report_pack_status"] == "completed"
    assert payload["diagnostics"]["errors"] == []


@pytest.mark.fast
def test_context_builder_backfills_attention_case_evidence_from_index() -> None:
    evidence_ref = EvidenceRef(
        ref_id="evidence://artifact/scheduler",
        kind="artifact",
        path="artifacts/task-1/sample-1/trials/trial_0001/agent/scheduler_result.json",
        mime_type="application/json",
        size_bytes=123,
        sha256="abc123",
        timestamp_iso="2026-05-14T00:00:00Z",
        task_id="task-1",
        sample_id="task-1:sample-1",
        trial_id="trial_0001",
        artifact_role="agent",
    )

    class Index:
        evidence_refs = [evidence_ref]
        diagnostics = None

    context = ReportContextBuilder().build(
        index=Index(),
        summary_payload={"sample_count": 1, "metrics": []},
        metrics=[],
        tasks=[{"task_id": "task-1", "sample_count": 1, "metrics": []}],
        runtime_health={"sample_count": 1, "completed_count": 0, "failed_count": 1, "aborted_count": 0},
        observability_health={"events_emitted_total": 0},
        generator_result=SummaryGeneratorResult(
            attention_cases=[
                {
                    "case_id": "task-1/sample-1",
                    "task_id": "task-1",
                    "sample_id": "sample-1",
                    "trial_id": "trial_0001",
                    "severity": "high",
                    "reason_codes": ["score.low"],
                    "summary": "Sample failed.",
                    "evidence_ref_ids": [],
                    "scoring": {
                        "frequency": 1.0,
                        "impact": "high",
                        "actionability": "medium",
                        "priority_score": 0.78,
                    },
                }
            ]
        ),
    )

    payload = context.to_dict()
    assert payload["attention_cases"][0]["evidence_ref_ids"] == ["evidence://artifact/scheduler"]
    assert payload["failure_clusters"][0]["representative_ref_ids"] == ["evidence://artifact/scheduler"]
    assert payload["diagnostics"]["errors"] == []
