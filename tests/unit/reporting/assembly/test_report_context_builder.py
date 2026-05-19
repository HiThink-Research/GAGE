from __future__ import annotations

import hashlib
import json

import pytest

from gage_eval.reporting.assembly.context_builder import ReportContextBuilder
from gage_eval.reporting.contracts import CaseDetails, EvidenceRef, SummaryGeneratorResult
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
def test_context_builder_injects_external_harness_metrics_from_samples() -> None:
    class Index:
        evidence_refs = []
        diagnostics = None

    context = ReportContextBuilder().build(
        index=Index(),
        summary_payload={"sample_count": 1, "metrics": []},
        metrics=[],
        tasks=[{"task_id": "tb2_one_case", "sample_count": 1, "metrics": []}],
        runtime_health={"sample_count": 1, "completed_count": 1, "failed_count": 0, "aborted_count": 0},
        observability_health={"events_emitted_total": 0},
        samples=[
            {
                "namespace": "task_tb2_one_case",
                "sample": {
                    "task_type": "external_harness.harbor",
                    "eval_result": {
                        "score": {"value": 0.0, "source_trial_id": "trial_0001"},
                        "resolved": {"value": False, "source_trial_id": "trial_0001"},
                    },
                },
            }
        ],
        generator_result=None,
    )

    payload = context.to_dict()
    assert [metric["metric_id"] for metric in payload["metrics"]] == [
        "harbor_score_mean",
        "harbor_resolve_rate",
    ]
    assert payload["headline"]["primary_metric"]["metric_id"] == "harbor_score_mean"
    assert payload["headline"]["score"] == 0.0
    task_metrics = payload["tasks"][0]["metrics"]
    assert [metric["metric_id"] for metric in task_metrics] == [
        "harbor_score_mean",
        "harbor_resolve_rate",
    ]
    assert task_metrics[0]["scope"] == "task"


@pytest.mark.fast
def test_context_builder_augments_runtime_health_from_task_status() -> None:
    class Index:
        evidence_refs = []
        diagnostics = None

    context = ReportContextBuilder().build(
        index=Index(),
        summary_payload={"sample_count": 0, "metrics": []},
        metrics=[],
        tasks=[{"task_id": "harbor", "execution": {"status": "failed"}, "sample_count": 0}],
        runtime_health={"sample_count": 0, "completed_count": 0, "failed_count": 0, "aborted_count": 0},
        observability_health={"events_emitted_total": 0},
        generator_result=None,
    )

    payload = context.to_dict()
    assert payload["runtime_health"]["task_failed_count"] == 1
    assert payload["headline"]["verdict"] == "failed"
    assert "1 task failed" in payload["headline"]["verdict_reason"]


@pytest.mark.fast
def test_context_builder_adds_completion_rate_metric_when_domain_metrics_are_missing() -> None:
    class Index:
        evidence_refs = []
        diagnostics = None

    context = ReportContextBuilder().build(
        index=Index(),
        summary_payload={"sample_count": 1, "metrics": []},
        metrics=[],
        tasks=[],
        runtime_health={"sample_count": 1, "completed_count": 0, "failed_count": 1, "aborted_count": 0},
        observability_health={"events_emitted_total": 0},
        generator_result=None,
    )

    payload = context.to_dict()
    assert payload["metrics"][0]["metric_id"] == "sample_completion_rate"
    assert payload["metrics"][0]["values"]["rate"] == "0.00000"
    assert payload["headline"]["primary_metric"]["metric_id"] == "sample_completion_rate"
    assert payload["headline"]["score"] == 0.0


@pytest.mark.fast
def test_context_builder_adds_task_success_rate_metric_for_pre_sample_task_failures() -> None:
    class Index:
        evidence_refs = []
        diagnostics = None

    context = ReportContextBuilder().build(
        index=Index(),
        summary_payload={"sample_count": 0, "metrics": []},
        metrics=[],
        tasks=[{"task_id": "harbor", "execution": {"status": "failed"}, "sample_count": 0}],
        runtime_health={"sample_count": 0, "completed_count": 0, "failed_count": 0, "aborted_count": 0},
        observability_health={"events_emitted_total": 0},
        generator_result=None,
    )

    payload = context.to_dict()
    assert payload["metrics"][0]["metric_id"] == "task_success_rate"
    assert payload["metrics"][0]["values"]["rate"] == "0.00000"
    assert payload["headline"]["primary_metric"]["metric_id"] == "task_success_rate"


@pytest.mark.fast
def test_context_builder_aggregates_task_scoped_metrics_for_run_headline() -> None:
    class Index:
        evidence_refs = []
        diagnostics = None

    context = ReportContextBuilder().build(
        index=Index(),
        summary_payload={"sample_count": 2, "metrics": []},
        metrics=[
            {
                "metric_id": "model_multi_choice_acc",
                "task_id": "task-a",
                "values": {"acc": "1.00000"},
                "raw_values": {"acc": 1.0},
            },
            {
                "metric_id": "model_multi_choice_acc",
                "task_id": "task-b",
                "values": {"acc": "0.00000"},
                "raw_values": {"acc": 0.0},
            },
        ],
        tasks=[],
        runtime_health={"sample_count": 2, "completed_count": 2, "failed_count": 0, "aborted_count": 0},
        observability_health={"events_emitted_total": 0},
        generator_result=None,
    )

    payload = context.to_dict()
    assert payload["metrics"] == [
        {
            "aggregation": "mean",
            "count": 2,
            "metric_id": "model_multi_choice_acc",
            "raw_values": {"acc": 0.5},
            "scope": "run",
            "source": "summary",
            "values": {"acc": "0.50000"},
        }
    ]
    assert payload["headline"]["primary_metric"]["metric_id"] == "model_multi_choice_acc"
    assert payload["headline"]["score"] == 0.5
    assert payload["methodology"]["metric_ids"] == ["model_multi_choice_acc"]
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


@pytest.mark.io
def test_context_builder_backfills_same_remote_media_url_for_multiple_samples(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    url = "https://example.test/assets/shared.png?token=secret"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    records = [
        {
            "sample_id": "sample-1",
            "task_id": "mmmu",
            "inputs": {"multi_modal_data": {"image": url}},
        },
        {
            "sample_id": "sample-2",
            "task_id": "mmmu",
            "inputs": {"multi_modal_data": {"image": url}},
        },
    ]
    (run_dir / "samples.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    index = ReportEvidenceReader().build_index(run_dir)

    context = ReportContextBuilder().build(
        index=index,
        summary_payload={"sample_count": 2, "metrics": []},
        metrics=[],
        tasks=[{"task_id": "mmmu", "sample_count": 2, "metrics": []}],
        runtime_health={"sample_count": 2, "completed_count": 0, "failed_count": 2, "aborted_count": 0},
        observability_health={"events_emitted_total": 0},
        generator_result=SummaryGeneratorResult(
            attention_cases=[
                {
                    "case_id": "mmmu/sample-1",
                    "task_id": "mmmu",
                    "sample_id": "sample-1",
                    "severity": "high",
                    "reason_codes": ["score.low"],
                    "summary": "Sample 1 failed.",
                    "evidence_ref_ids": [],
                    "scoring": {
                        "frequency": 0.5,
                        "impact": "high",
                        "actionability": "medium",
                        "priority_score": 0.64,
                    },
                },
                {
                    "case_id": "mmmu/sample-2",
                    "task_id": "mmmu",
                    "sample_id": "sample-2",
                    "severity": "high",
                    "reason_codes": ["score.low"],
                    "summary": "Sample 2 failed.",
                    "evidence_ref_ids": [],
                    "scoring": {
                        "frequency": 0.5,
                        "impact": "high",
                        "actionability": "medium",
                        "priority_score": 0.64,
                    },
                },
            ]
        ),
    )

    refs_by_sample = {ref.sample_id: ref for ref in context.evidence_refs}
    assert set(refs_by_sample) == {"sample-1", "sample-2"}
    assert {ref.path for ref in refs_by_sample.values()} == {f"external://sha256/{digest}"}
    assert context.attention_cases[0].evidence_ref_ids == [refs_by_sample["sample-1"].ref_id]
    assert context.attention_cases[1].evidence_ref_ids == [refs_by_sample["sample-2"].ref_id]
    assert context.diagnostics["errors"] == []


@pytest.mark.fast
def test_context_builder_auto_generates_case_details_from_attention_case_evidence() -> None:
    scheduler_ref = _evidence_ref(
        "evidence://artifact/scheduler",
        "artifacts/task-1/sample-1/trials/trial_0001/agent/scheduler_result.json",
    )
    sample_ref = _evidence_ref(
        "evidence://sample/task-1/sample-1",
        "samples/task-1/sample-1/sample_record.json",
    )

    class Index:
        evidence_refs = [scheduler_ref, sample_ref]
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
                _attention_case(
                    evidence_ref_ids=[
                        "evidence://artifact/scheduler",
                        "evidence://sample/task-1/sample-1",
                    ]
                )
            ],
        ),
    )

    payload = context.to_dict()
    detail = payload["case_details"]["task-1/sample-1"]
    assert detail["evidence_ref_ids"] == [
        "evidence://artifact/scheduler",
        "evidence://sample/task-1/sample-1",
    ]
    assert detail["artifact_preview_ref_ids"] == [
        "evidence://artifact/scheduler",
        "evidence://sample/task-1/sample-1",
    ]
    assert detail["scoring_breakdown"] == {
        "frequency": 1.0,
        "impact": "high",
        "actionability": "medium",
        "priority_score": 0.78,
    }
    assert "case_id" not in detail


@pytest.mark.fast
def test_context_builder_keeps_existing_case_details() -> None:
    trace_ref = _evidence_ref(
        "evidence://trial/trace",
        "artifacts/task-1/sample-1/trials/trial_0001/agent/trace.jsonl",
    )

    class Index:
        evidence_refs = [trace_ref]
        diagnostics = None

    context = ReportContextBuilder().build(
        index=Index(),
        summary_payload={"sample_count": 1, "metrics": []},
        metrics=[],
        tasks=[{"task_id": "task-1", "sample_count": 1, "metrics": []}],
        runtime_health={"sample_count": 1, "completed_count": 0, "failed_count": 1, "aborted_count": 0},
        observability_health={"events_emitted_total": 0},
        generator_result=SummaryGeneratorResult(
            attention_cases=[_attention_case(evidence_ref_ids=["evidence://trial/trace"])],
            case_details={
                "task-1/sample-1": CaseDetails(
                    message_history_preview=[{"role": "assistant", "content": "kept"}],
                    scoring_breakdown={"generator": "kept"},
                    artifact_preview_ref_ids=["evidence://trial/trace"],
                    evidence_ref_ids=["evidence://trial/trace"],
                    full_trace_ref_id="evidence://trial/trace",
                )
            },
        ),
    )

    detail = context.to_dict()["case_details"]["task-1/sample-1"]
    assert detail["message_history_preview"] == [{"role": "assistant", "content": "kept"}]
    assert detail["scoring_breakdown"] == {"generator": "kept"}
    assert detail["artifact_preview_ref_ids"] == ["evidence://trial/trace"]


@pytest.mark.fast
def test_context_builder_selects_trace_jsonl_ref_for_generated_case_detail() -> None:
    scheduler_ref = _evidence_ref(
        "evidence://artifact/scheduler",
        "artifacts/task-1/sample-1/trials/trial_0001/agent/scheduler_result.json",
    )
    trace_ref = _evidence_ref(
        "evidence://trial/trace",
        "artifacts/task-1/sample-1/trials/trial_0001/agent/trace.jsonl",
    )

    class Index:
        evidence_refs = [scheduler_ref, trace_ref]
        diagnostics = None

    context = ReportContextBuilder().build(
        index=Index(),
        summary_payload={"sample_count": 1, "metrics": []},
        metrics=[],
        tasks=[{"task_id": "task-1", "sample_count": 1, "metrics": []}],
        runtime_health={"sample_count": 1, "completed_count": 0, "failed_count": 1, "aborted_count": 0},
        observability_health={"events_emitted_total": 0},
        generator_result=SummaryGeneratorResult(attention_cases=[_attention_case(evidence_ref_ids=[])]),
    )

    detail = context.to_dict()["case_details"]["task-1/sample-1"]
    assert detail["evidence_ref_ids"] == ["evidence://artifact/scheduler", "evidence://trial/trace"]
    assert detail["full_trace_ref_id"] == "evidence://trial/trace"


@pytest.mark.fast
def test_context_builder_does_not_match_short_sample_id_by_path_substring() -> None:
    unrelated_ref = EvidenceRef(
        ref_id="evidence://artifact/unrelated",
        kind="artifact",
        path="artifacts/task-1/sample-10/trials/trial_0001/agent/scheduler_result.json",
        mime_type="application/json",
        size_bytes=123,
        sha256="abc123",
        timestamp_iso="2026-05-14T00:00:00Z",
        task_id="task-1",
        trial_id="trial_0001",
        artifact_role="agent",
    )

    class Index:
        evidence_refs = [unrelated_ref]
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
                    "case_id": "task-1/1",
                    "task_id": "task-1",
                    "sample_id": "1",
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

    assert context.attention_cases[0].evidence_ref_ids == []


def _attention_case(*, evidence_ref_ids: list[str]) -> dict:
    return {
        "case_id": "task-1/sample-1",
        "task_id": "task-1",
        "sample_id": "sample-1",
        "trial_id": "trial_0001",
        "severity": "high",
        "reason_codes": ["score.low"],
        "summary": "Sample failed.",
        "evidence_ref_ids": evidence_ref_ids,
        "scoring": {
            "frequency": 1.0,
            "impact": "high",
            "actionability": "medium",
            "priority_score": 0.78,
        },
    }


def _evidence_ref(ref_id: str, path: str) -> EvidenceRef:
    return EvidenceRef(
        ref_id=ref_id,
        kind="artifact",
        path=path,
        mime_type="application/json",
        size_bytes=123,
        sha256="abc123",
        timestamp_iso="2026-05-14T00:00:00Z",
        task_id="task-1",
        sample_id="task-1:sample-1",
        trial_id="trial_0001",
        artifact_role="agent",
    )
