from __future__ import annotations

import json

import pytest

from gage_eval.evaluation.cache import EvalCache
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.report import ReportStep


@pytest.mark.io
def test_report_step_finalize_writes_summary_and_report_pack(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_REPORT_PACK", "1")
    cache = EvalCache(base_dir=tmp_path, run_id="finalize-e2e")
    cache.write_sample("sample", {"sample": {"id": "sample"}, "status": "completed"})

    payload = ReportStep(auto_eval_step=None, cache_store=cache).finalize(ObservabilityTrace(run_id="finalize-e2e"))

    assert payload["sample_count"] == 1
    assert (tmp_path / "finalize-e2e" / "summary.json").exists()
    assert (tmp_path / "finalize-e2e" / "report_pack" / "report.html").exists()
    summary = json.loads((tmp_path / "finalize-e2e" / "summary.json").read_text(encoding="utf-8"))
    assert summary["report_pack"]["status"] == "completed"
    diagnostics = json.loads((tmp_path / "finalize-e2e" / "report_pack" / "diagnostics.json").read_text(encoding="utf-8"))
    warning_codes = {item["code"] for item in diagnostics.get("warnings", [])}
    assert "report_pack.summary_missing" not in warning_codes


@pytest.mark.io
def test_report_step_validates_profile_refs_after_profile_injection(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_REPORT_PACK", "1")
    cache = EvalCache(base_dir=tmp_path, run_id="finalize-profile-validation")
    cache.write_sample("sample", {"sample": {"id": "sample"}, "status": "completed"})

    class _InvalidProfileBuilder:
        def build(self, index):  # noqa: ANN001
            return (
                {
                    "agent": {
                        "profile_version": "gage.scenario.agent.v1",
                        "trial_count": 1,
                        "failed_trial_count": 0,
                        "representative_ref_ids": ["evidence://artifact/missing"],
                    }
                },
                {},
            )

    monkeypatch.setattr(
        "gage_eval.pipeline.steps.report.ScenarioProfileBuilder",
        lambda: _InvalidProfileBuilder(),
    )

    payload = ReportStep(auto_eval_step=None, cache_store=cache).finalize(
        ObservabilityTrace(run_id="finalize-profile-validation")
    )

    assert payload["report_pack"]["status"] == "degraded"
    diagnostics = json.loads(
        (tmp_path / "finalize-profile-validation" / "report_pack" / "diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostics["report_pack_status"] == "degraded"
    assert any(
        item.get("code") == "report_context.evidence_ref_missing"
        and item.get("path") == "scenario_profiles.agent.representative_ref_ids[0]"
        for item in diagnostics.get("errors", [])
    )


@pytest.mark.io
def test_report_step_marks_pre_sample_task_batch_failure_in_headline(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_REPORT_PACK", "1")
    cache = EvalCache(base_dir=tmp_path, run_id="task-batch-failed")
    task = {
        "task_id": "external-task",
        "dataset_id": "external-dataset",
        "sample_count": 0,
        "metrics": [],
        "execution": {
            "status": "failed",
            "failed_step": "harbor_result",
            "failure": {
                "error_type": "ExternalHarnessParseError",
                "message": "harbor.launcher_failed",
            },
        },
    }

    payload = ReportStep(auto_eval_step=None, cache_store=cache).finalize(
        ObservabilityTrace(run_id="task-batch-failed"),
        tasks=[task],
    )
    context = json.loads(
        (tmp_path / "task-batch-failed" / "report_pack" / "report_context.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["runtime_health"]["task_failed_count"] == 1
    assert context["runtime_health"]["task_failed_count"] == 1
    assert context["headline"]["verdict"] == "failed"
    assert "1 task failed" in context["headline"]["verdict_reason"]
