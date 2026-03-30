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
