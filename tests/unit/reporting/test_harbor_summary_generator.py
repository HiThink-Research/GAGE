from __future__ import annotations

import json

import pytest

from gage_eval.evaluation.cache import EvalCache
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.report import ReportStep, _record_task_id
from gage_eval.registry import import_asset_from_manifest, registry
from gage_eval.reporting.summary_generators.harbor import HarborSummaryGenerator


@pytest.mark.fast
def test_summary_generator_consumes_imported_harbor_aggregate(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="harbor-summary")
    _write_harbor_sample(cache)

    payload = HarborSummaryGenerator().generate(cache)

    summary = payload["external_harness"]["harbor"]
    assert summary["task_ids"] == ["tb2_one_case"]
    assert summary["dataset_ids"] == ["terminal_bench_2_0"]
    assert summary["sample_count"] == 1
    assert summary["trial_count"] == 2
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["skipped"] == 1
    assert summary["harbor_resolve_rate"] == 1.0
    assert summary["harbor_score_mean"] == 0.75
    assert summary["external_trial_pass_hat_k"]["pass_hat@1"] == 0.5
    assert summary["failure_rollup"]["failure_codes"]["harbor.trial_exception"] == 1


@pytest.mark.fast
def test_report_output_includes_external_harness_harbor_section(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="harbor-report")
    cache.set_metadata("summary_generators", ["harbor_summary"])
    _write_harbor_sample(cache)
    report = ReportStep(auto_eval_step=None, cache_store=cache)

    payload = report.finalize(ObservabilityTrace(run_id="harbor-report"))

    assert payload["sample_count"] == 1
    assert payload["external_harness"]["harbor"]["sample_count"] == 1
    assert payload["external_harness"]["harbor"]["harbor_score_mean"] == 0.75
    assert [metric["metric_id"] for metric in payload["metrics"]] == [
        "harbor_score_mean",
        "harbor_resolve_rate",
    ]
    assert payload["metrics"][0]["values"]["mean"] == "0.75000"
    assert payload["metrics"][0]["raw_values"]["mean"] == 0.75
    assert payload["metrics"][1]["values"]["rate"] == "1.00000"
    assert payload["metrics"][1]["raw_values"]["rate"] == 1.0


@pytest.mark.fast
def test_report_output_includes_external_harness_harbor_task_metrics(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="harbor-report-task-metrics")
    cache.set_metadata("summary_generators", ["harbor_summary"])
    _write_harbor_sample(cache)
    report = ReportStep(auto_eval_step=None, cache_store=cache)

    payload = report.finalize(
        ObservabilityTrace(run_id="harbor-report-task-metrics"),
        tasks=[{"task_id": "tb2_one_case"}],
    )

    task_metrics = payload["tasks"][0]["metrics"]
    assert [metric["metric_id"] for metric in task_metrics] == [
        "harbor_score_mean",
        "harbor_resolve_rate",
    ]
    assert task_metrics[0]["scope"] == "task"
    assert task_metrics[0]["task_id"] == "tb2_one_case"


@pytest.mark.fast
def test_external_harness_task_metric_task_id_ignores_global_namespace() -> None:
    assert _record_task_id({"namespace": "task_global"}) is None
    assert _record_task_id({"namespace": "task/tb2_one_case"}) == "tb2_one_case"
    assert _record_task_id({"namespace": "task_tb2_one_case"}) == "tb2_one_case"


@pytest.mark.fast
def test_summary_generator_reports_cancelled_harbor_trials_as_aborted(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="harbor-cancelled-summary")
    _write_cancelled_harbor_sample(cache)

    payload = HarborSummaryGenerator().generate(cache)

    summary = payload["external_harness"]["harbor"]
    assert summary["sample_count"] == 1
    assert summary["trial_count"] == 1
    assert summary["completed"] == 0
    assert summary["aborted"] == 1
    assert summary["failure_rollup"]["status_counts"]["aborted"] == 1
    assert summary["failure_rollup"]["failure_codes"]["external_harness.cancelled.subprocess_aborted"] == 1


@pytest.mark.fast
def test_user_facing_summary_does_not_expose_launcher_implementation_detail(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="harbor-summary-clean")
    _write_harbor_sample(cache)

    payload = HarborSummaryGenerator().generate(cache)
    serialized = json.dumps(payload, sort_keys=True)

    assert "python_subprocess" not in serialized
    assert "launcher_argv" not in serialized
    assert "job_config_path" not in serialized
    assert "harbor_job_result.json" in serialized
    assert "harbor_raw_result.json" in serialized


@pytest.mark.fast
def test_harbor_summary_generator_is_registered_from_core_manifest() -> None:
    import_asset_from_manifest("summary_generators", "harbor_summary", registry=registry)

    assert registry.get("summary_generators", "harbor_summary") is HarborSummaryGenerator


@pytest.mark.fast
def test_harbor_v2_context_uses_overview_section_id() -> None:
    result = HarborSummaryGenerator().generate(
        {
            "samples": [
                {
                    "sample": {
                        "task_type": "external_harness.harbor",
                        "dataset_id": "terminal_bench_2_0",
                        "eval_result": {
                            "harbor_resolve_rate": 1.0,
                            "harbor_score_mean": 0.75,
                        },
                    },
                    "trial_results": [
                        {
                            "trial_id": "trial_0001",
                            "status": "completed",
                            "verifier_result": {"score": 0.75, "passed": True},
                        }
                    ],
                }
            ]
        }
    )

    assert result.generator_id == "harbor_summary"
    assert result.summary_sections[0]["generator_id"] == "harbor_summary"
    assert result.summary_sections[0]["section_id"] == "overview"


def _write_harbor_sample(cache: EvalCache) -> None:
    cache.write_sample(
        "gpt2-codegolf",
        {
            "sample": {
                "id": "gpt2-codegolf",
                "task_type": "external_harness.harbor",
                "dataset_id": "terminal_bench_2_0",
                "metadata": {
                    "_harness": {
                        "kit_id": "harbor",
                        "job_name": "gage_tb2",
                        "harbor_task_key": "gpt2-codegolf",
                    },
                    "dataset_id": "terminal_bench_2_0",
                },
                "eval_result": {
                    "harbor_resolve_rate": 1.0,
                    "harbor_score_mean": 0.75,
                    "external_trial_pass_values": [True, None],
                    "external_trial_metric_projection": {
                        "trial_ids": ["trial_0001", "trial_0002"],
                        "skipped_failed_trials": [
                            {"trial_id": "trial_0002", "failure_code": "harbor.trial_exception"}
                        ],
                    },
                },
            },
            "judge_output": {
                "harbor_resolve_rate": 1.0,
                "harbor_score_mean": 0.75,
                "external_trial_pass_values": [True, None],
            },
            "trial_results": [
                {
                    "trial_id": "trial_0001",
                    "status": "completed",
                    "verifier_result": {"score": 0.75, "passed": True, "resolved": True},
                    "failure": None,
                },
                {
                    "trial_id": "trial_0002",
                    "status": "failed",
                    "verifier_result": {"score": None, "passed": None, "resolved": None},
                    "failure": {"failure_code": "harbor.trial_exception"},
                },
            ],
            "aggregate_result": {
                "trial_count": 2,
                "completed_trial_count": 1,
                "failed_trial_count": 1,
                "metric_projection": {
                    "skipped_failed_trials": [
                        {"trial_id": "trial_0002", "failure_code": "harbor.trial_exception"}
                    ],
                },
                "failure_rollup": {
                    "status_counts": {"completed": 1, "failed": 1},
                    "failure_codes": {"harbor.trial_exception": 1},
                    "failure_domains": {"external_harness": 1},
                },
            },
            "artifact_refs": [
                {
                    "owner": "infra",
                    "name": "harbor_job_result.json",
                    "path": "artifacts/tb2_one_case/gpt2-codegolf/infra/harbor_job_result.json",
                    "mime_type": "application/json",
                    "size_bytes": 2,
                    "sha256": "0" * 64,
                },
                {
                    "owner": "infra",
                    "name": "harbor_raw_result.json",
                    "path": "artifacts/tb2_one_case/gpt2-codegolf/trials/trial_0001/infra/harbor_raw_result.json",
                    "mime_type": "application/json",
                    "size_bytes": 2,
                    "sha256": "0" * 64,
                },
            ],
        },
        namespace="task/tb2_one_case",
    )


def _write_cancelled_harbor_sample(cache: EvalCache) -> None:
    cache.write_sample(
        "gpt2-codegolf",
        {
            "sample": {
                "id": "gpt2-codegolf",
                "task_type": "external_harness.harbor",
                "dataset_id": "terminal_bench_2_0",
                "metadata": {
                    "_harness": {
                        "kit_id": "harbor",
                        "job_name": "gage_tb2",
                        "harbor_task_key": "gpt2-codegolf",
                    },
                    "dataset_id": "terminal_bench_2_0",
                },
                "eval_result": {
                    "harbor_resolve_rate": None,
                    "harbor_score_mean": None,
                    "external_trial_pass_values": [None],
                },
            },
            "trial_results": [
                {
                    "trial_id": "trial_0001",
                    "status": "aborted",
                    "verifier_result": {"score": None, "passed": None, "resolved": None},
                    "failure": {
                        "failure_code": "external_harness.cancelled.subprocess_aborted",
                        "failure_domain": "external_harness",
                    },
                },
            ],
            "aggregate_result": {
                "trial_count": 1,
                "completed_trial_count": 0,
                "failed_trial_count": 1,
                "failure_rollup": {
                    "status_counts": {"aborted": 1},
                    "failure_codes": {"external_harness.cancelled.subprocess_aborted": 1},
                    "failure_domains": {"external_harness": 1},
                },
            },
        },
        namespace="task/tb2_one_case",
    )
