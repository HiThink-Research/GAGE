from __future__ import annotations

from types import SimpleNamespace

import pytest

from gage_eval.evaluation.cache import EvalCache
from gage_eval.evaluation.pipeline import PipelineRuntime
from gage_eval.evaluation.runtime_builder import TaskOrchestratorRuntime
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.report import ReportStep
from gage_eval.reporting.recorders import InMemoryRecorder


class _ReportStepStub:
    def __init__(self, sample_count: int) -> None:
        self._sample_count = sample_count

    def get_sample_count(self) -> int:
        return self._sample_count


@pytest.mark.fast
def test_report_step_exposes_sample_count_from_cache(tmp_path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="report-step-contract")
    report = ReportStep(auto_eval_step=None, cache_store=cache)

    cache.write_sample("sample-1", {"value": 1})
    cache.write_sample("sample-2", {"value": 2})

    assert report.get_sample_count() == 2


@pytest.mark.fast
def test_report_summary_includes_runtime_health_counts(tmp_path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="runtime-health")
    report = ReportStep(auto_eval_step=None, cache_store=cache)
    trace = ObservabilityTrace(
        recorder=InMemoryRecorder(run_id="runtime-health"),
        run_id="runtime-health",
    )
    cache.write_sample(
        "sample-1",
        {
            "model_output": {
                "runtime_judge_outcome": {
                    "verifier_input": {
                        "scheduler_result": {
                            "status": "failed",
                            "failure_code": "client_execution.tool_retry_budget_exhausted",
                        }
                    },
                    "judge_output": {
                        "status": "skipped",
                        "failure_code": "verifier.skipped_due_to_scheduler_failure",
                    },
                }
            },
            "judge_output": {
                "status": "skipped",
                "failure_code": "verifier.skipped_due_to_scheduler_failure",
            },
        },
    )

    payload = report.finalize(trace)

    assert payload["runtime_health"] == {
        "sample_count": 1,
        "completed_count": 0,
        "failed_count": 1,
        "aborted_count": 0,
        "verifier_skipped_count": 1,
        "scheduler_failed_count": 1,
    }


@pytest.mark.fast
def test_pipeline_runtime_resolves_sample_count_via_report_public_api() -> None:
    runtime = PipelineRuntime(
        sample_loop=SimpleNamespace(processed_count=2, shutdown=lambda: None),
        task_planner=SimpleNamespace(),
        role_manager=SimpleNamespace(shutdown=lambda: None),
        trace=ObservabilityTrace(
            recorder=InMemoryRecorder(run_id="pipeline-report-contract"),
            run_id="pipeline-report-contract",
        ),
        report_step=_ReportStepStub(7),
    )

    assert runtime._resolve_sample_count() == 7


@pytest.mark.fast
def test_task_orchestrator_runtime_resolves_sample_count_via_report_public_api() -> None:
    runtime = TaskOrchestratorRuntime(
        tasks=[],
        role_manager=SimpleNamespace(shutdown=lambda: None),
        trace=ObservabilityTrace(
            recorder=InMemoryRecorder(run_id="task-report-contract"),
            run_id="task-report-contract",
        ),
        report_step=_ReportStepStub(11),
    )

    assert runtime._resolve_sample_count() == 11
