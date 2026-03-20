from __future__ import annotations

import json

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

    def __init__(self, fail_on: str = "s0") -> None:
        self.adapter_id = "dut"
        self.fail_on = fail_on

    def clone_for_sample(self):
        return self

    def invoke(self, payload, state=None):
        sample = payload.get("sample", {})
        if sample.get("id") == self.fail_on:
            raise RuntimeError("boom")
        msg = f"ok-{sample.get('id')}"
        return {
            "answer": msg,
            "message": {"role": "assistant", "content": [{"type": "text", "text": msg}]},
        }


class _EchoRole(_FailingRole):
    def __init__(self) -> None:
        super().__init__(fail_on="never")


@pytest.mark.fast
def test_pipeline_runtime_writes_execution_summary_on_abort(tmp_path) -> None:
    samples = [
        {
            "id": f"s{idx}",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "choices": [],
        }
        for idx in range(3)
    ]
    sample_loop = SampleLoop(samples, concurrency=1, max_inflight=2, failure_policy="fail_fast")
    planner = TaskPlanner()
    planner.configure_custom_steps(
        [
            {"step": "inference", "adapter_id": "dut"},
            {"step": "auto_eval"},
        ]
    )
    role_manager = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=1)]))
    role_manager.register_role_adapter("dut", _FailingRole())
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="pipeline-abort"))
    cache = EvalCache(base_dir=tmp_path, run_id=trace.run_id)
    runtime = PipelineRuntime(
        sample_loop=sample_loop,
        task_planner=planner,
        role_manager=role_manager,
        trace=trace,
        report_step=ReportStep(auto_eval_step=None, cache_store=cache),
    )

    with pytest.raises(SampleLoopExecutionError) as exc_info:
        runtime.run()

    summary = json.loads((cache.run_dir / "summary.json").read_text(encoding="utf-8"))

    assert exc_info.value.outcome.status == "aborted"
    assert summary["execution"]["status"] == "aborted"
    assert summary["execution"]["failed_sample_id"] == "s0"
    assert summary["execution"]["cancelled_samples"] >= 1


@pytest.mark.fast
def test_pipeline_runtime_interval_writer_close_patches_final_fsync(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_ENABLE_BUFFERED_WRITER", "1")
    monkeypatch.setenv("GAGE_EVAL_BUFFER_DURABILITY_POLICY", "interval")
    monkeypatch.setenv("GAGE_EVAL_BUFFER_FSYNC_EVERY_FLUSHES", "99")
    monkeypatch.setenv("GAGE_EVAL_BUFFER_FSYNC_EVERY_S", "999")
    samples = [
        {
            "id": f"s{idx}",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "choices": [],
        }
        for idx in range(2)
    ]
    sample_loop = SampleLoop(samples, concurrency=1, max_inflight=2, failure_policy="fail_fast")
    planner = TaskPlanner()
    planner.configure_custom_steps(
        [
            {"step": "inference", "adapter_id": "dut"},
            {"step": "auto_eval"},
        ]
    )
    role_manager = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=1)]))
    role_manager.register_role_adapter("dut", _EchoRole())
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="pipeline-buffered"))
    cache = EvalCache(base_dir=tmp_path, run_id=trace.run_id)
    runtime = PipelineRuntime(
        sample_loop=sample_loop,
        task_planner=planner,
        role_manager=role_manager,
        trace=trace,
        report_step=ReportStep(auto_eval_step=None, cache_store=cache),
    )
    cache.write_sample("seed", {"value": 1}, namespace="default")

    runtime.run()

    summary = json.loads((cache.run_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["buffered_writer_durability_policy"] == "interval"
    assert summary["buffered_writer_flush_count"] > 0
    assert summary["buffered_writer_fsync_count"] > 0
