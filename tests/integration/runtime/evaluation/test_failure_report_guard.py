from __future__ import annotations

import pytest

from gage_eval.evaluation.cache import EvalCache
from gage_eval.evaluation.execution_controller import SampleLoopExecutionError
from gage_eval.evaluation.pipeline import PipelineRuntime
from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.evaluation.task_planner import TaskPlanner
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.report import ReportStep
from gage_eval.reporting.recorders import InMemoryRecorder
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager


class _FailingRole:
    role_type = "dut_model"
    resource_requirement = {}
    backend = None

    def __init__(self) -> None:
        self.adapter_id = "dut"

    def clone_for_sample(self):
        return self

    def invoke(self, payload, state=None):
        raise RuntimeError("boom")


@pytest.mark.fast
def test_report_finalize_failure_does_not_mask_primary_sample_error(tmp_path) -> None:
    sample_loop = SampleLoop(
        [
            {
                "id": "s0",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                "choices": [],
            }
        ],
        concurrency=1,
        failure_policy="fail_fast",
    )
    planner = TaskPlanner()
    planner.configure_custom_steps([{"step": "inference", "adapter_id": "dut"}])
    role_manager = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=1)]))
    role_manager.register_role_adapter("dut", _FailingRole())
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="report-guard"))
    cache = EvalCache(base_dir=tmp_path, run_id=trace.run_id)
    report_step = ReportStep(auto_eval_step=None, cache_store=cache)
    runtime = PipelineRuntime(
        sample_loop=sample_loop,
        task_planner=planner,
        role_manager=role_manager,
        trace=trace,
        report_step=report_step,
    )

    def fail_finalize(*args, **kwargs):
        raise RuntimeError("report failed")

    report_step.finalize = fail_finalize  # type: ignore[assignment]

    with pytest.raises(SampleLoopExecutionError, match="boom") as exc_info:
        runtime.run()

    assert exc_info.value.outcome.failed_sample_id == "s0"
    assert any(event["event"] == "report_finalize_failed_after_abort" for event in trace.events)
