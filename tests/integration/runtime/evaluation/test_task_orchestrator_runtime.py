import json
from pathlib import Path

import pytest
from loguru import logger

from gage_eval.config.pipeline_config import (
    CustomPipelineStep,
    DatasetSpec,
    PipelineConfig,
    RoleAdapterSpec,
    TaskSpec,
)
from gage_eval.evaluation.cache import EvalCache
from gage_eval.evaluation.runtime_builder import TaskOrchestratorRuntime, _prepare_task_entries, _record_config_metadata
from gage_eval.observability.config import ObservabilityConfig, get_observability_config, set_observability_config
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder, RecorderBase, ResilientRecorder, TraceEvent
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager
from gage_eval.assets.datasets.manager import DataManager, DataSource
from gage_eval.pipeline.steps.report import ReportStep
from gage_eval.metrics import MetricRegistry
from gage_eval.evaluation.task_plan import build_task_plan_specs


@pytest.fixture(autouse=True)
def _enable_observability():
    original = get_observability_config()
    set_observability_config(ObservabilityConfig(enabled=True))
    try:
        yield
    finally:
        set_observability_config(original)


class _EchoRole:
    def __init__(self, adapter_id: str, role_type: str, **kwargs) -> None:
        self.adapter_id = adapter_id
        self.role_type = role_type
        self.resource_requirement = kwargs.get("resource_requirement", {})
        self.backend = kwargs.get("backend")

    def clone_for_sample(self):
        return self

    def invoke(self, payload, state=None):
        sample = payload.get("sample", {})
        msg = f"echo-{sample.get('id')}"
        return {"answer": msg, "message": {"role": "assistant", "content": [{"type": "text", "text": msg}]}}


class _LoggingRole(_EchoRole):
    def invoke(self, payload, state=None):
        sample = payload.get("sample", {})
        logger.bind(stage="runtime_test_log").info("logging-role invoked sample={}", sample.get("id"))
        return super().invoke(payload, state=state)


class _FailingRole(_EchoRole):
    def invoke(self, payload, state=None):
        sample = payload.get("sample", {})
        if sample.get("id") == "s0":
            raise RuntimeError("boom")
        return super().invoke(payload, state=state)


def _make_runtime(
    tmp_path: Path,
    sample_count: int = 4,
    trace: ObservabilityTrace | None = None,
    role_cls=_EchoRole,
):
    dataset = DataSource(
        dataset_id="ds",
        records=[
            {
                "id": f"s{idx}",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                "choices": [],
            }
            for idx in range(sample_count)
        ],
        metadata={"path": str(tmp_path / "ds.jsonl")},
    )
    dm = DataManager()
    dm.register_source(dataset)

    config = PipelineConfig(
        datasets=(DatasetSpec(dataset_id="ds", loader="jsonl"),),
        role_adapters=(RoleAdapterSpec(adapter_id="dut", role_type="dut_model", class_path="tests.integration.runtime.evaluation.test_task_orchestrator_runtime._EchoRole"),),
        tasks=(
            TaskSpec(task_id="t1", dataset_id="ds", steps=(CustomPipelineStep(step_type="inference", adapter_id="dut"),)),
            TaskSpec(task_id="t2", dataset_id="ds", steps=(CustomPipelineStep(step_type="inference", adapter_id="dut"),), max_samples=2),
        ),
    )
    plans = build_task_plan_specs(config)
    trace = trace or ObservabilityTrace(
        recorder=InMemoryRecorder(run_id="runtime-test"),
        run_id="runtime-test",
    )
    cache = EvalCache(base_dir=tmp_path, run_id=trace.run_id)
    report_step = ReportStep(auto_eval_step=None, cache_store=cache)
    entries = _prepare_task_entries(
        task_plans=plans,
        config=config,
        data_manager=dm,
        datasets={"ds": dataset},
        trace=trace,
        cache_store=cache,
        resource_profile=ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=2)]),
        sandbox_profiles={},
    )

    rm = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=2)]))
    rm.register_role_adapter("dut", role_cls("dut", "dut_model"))

    runtime = TaskOrchestratorRuntime(entries, rm, trace, report_step)
    _record_config_metadata(config, cache, trace=trace)
    return runtime, trace, cache, entries


class _FlushFailRecorder(RecorderBase):
    def __init__(self, run_id: str) -> None:
        super().__init__(run_id, min_flush_events=10_000, min_flush_seconds=10_000.0)

    def _flush_events_internal(self, events: tuple[TraceEvent, ...] | list[TraceEvent]) -> None:
        raise RuntimeError("flush failed")


def _make_runtime_with_steps(tmp_path: Path, steps, sample_count: int = 2):
    dataset = DataSource(
        dataset_id="ds",
        records=[
            {
                "id": f"s{idx}",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                "choices": [],
            }
            for idx in range(sample_count)
        ],
        metadata={"path": str(tmp_path / "ds.jsonl")},
    )
    dm = DataManager()
    dm.register_source(dataset)

    config = PipelineConfig(
        datasets=(DatasetSpec(dataset_id="ds", loader="jsonl"),),
        role_adapters=(RoleAdapterSpec(adapter_id="dut", role_type="dut_model", class_path="tests.integration.runtime.evaluation.test_task_orchestrator_runtime._EchoRole"),),
        tasks=(TaskSpec(task_id="t1", dataset_id="ds", steps=steps),),
    )
    plans = build_task_plan_specs(config)
    trace = ObservabilityTrace(
        recorder=InMemoryRecorder(run_id="runtime-auto-eval-test"),
        run_id="runtime-auto-eval-test",
    )
    cache = EvalCache(base_dir=tmp_path, run_id=trace.run_id)
    report_step = ReportStep(auto_eval_step=None, cache_store=cache)
    entries = _prepare_task_entries(
        task_plans=plans,
        config=config,
        data_manager=dm,
        datasets={"ds": dataset},
        trace=trace,
        cache_store=cache,
        resource_profile=ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=2)]),
        sandbox_profiles={},
    )

    rm = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=2)]))
    rm.register_role_adapter("dut", _EchoRole("dut", "dut_model"))

    runtime = TaskOrchestratorRuntime(entries, rm, trace, report_step)
    _record_config_metadata(config, cache, trace=trace)
    return runtime, cache


def test_task_orchestrator_runs_tasks(tmp_path: Path):
    runtime, trace, cache, entries = _make_runtime(tmp_path)

    runtime.run()

    task_starts = [e for e in trace.events if e["event"] == "task_start"]
    task_ends = [e for e in trace.events if e["event"] == "task_end"]
    assert len(task_starts) == 2
    assert len(task_ends) == 2
    assert all(entry.sample_loop.processed_count > 0 for entry in entries)
    assert cache.get_metadata("run_identity")["run_id"] == trace.run_id
    summary = cache.run_dir / "summary.json"
    assert summary.exists()
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["observability_closed_cleanly"] is True
    assert payload["observability_close_mode"] == "drain"
    assert payload["observability_close_remaining_queue"] == 0


def test_task_orchestrator_records_timings(tmp_path: Path):
    runtime, trace, cache, entries = _make_runtime(tmp_path, sample_count=2)
    runtime.run()

    timings = cache._timings
    assert "inference_s" in timings and timings["inference_s"] >= 0.0


def test_auto_eval_writes_samples_without_metrics(tmp_path: Path):
    steps = (
        CustomPipelineStep(step_type="inference", adapter_id="dut"),
        CustomPipelineStep(step_type="auto_eval"),
    )
    runtime, cache = _make_runtime_with_steps(tmp_path, steps)

    runtime.run()

    samples_jsonl = cache.run_dir / "samples.jsonl"
    assert samples_jsonl.exists()
    assert samples_jsonl.read_text(encoding="utf-8").strip()


def test_task_orchestrator_survives_trace_flush_failure(tmp_path: Path):
    fallback = InMemoryRecorder(run_id="runtime-trace-fallback")
    trace = ObservabilityTrace(
        recorder=ResilientRecorder(_FlushFailRecorder(run_id="runtime-trace"), fallback=fallback),
        run_id="runtime-trace",
    )
    runtime, trace, cache, entries = _make_runtime(tmp_path, sample_count=2, trace=trace)

    runtime.run()

    health = trace.health_snapshot()
    summary = cache.run_dir / "summary.json"

    assert summary.exists()
    assert health["observability_degraded"] is True
    assert health["observability_mode"] == "fallback"
    assert health["backlog_events"] == 0
    assert fallback.buffered_events()


def test_task_orchestrator_routes_worker_logs_into_trace(tmp_path: Path):
    runtime, trace, cache, entries = _make_runtime(tmp_path, sample_count=2)
    for entry in entries:
        entry.sample_loop.register_hook(
            lambda sample, task_id=entry.task_id: logger.bind(stage="runtime_test_log").info(
                "hook-log task={} sample={}",
                task_id,
                sample.get("id"),
            )
        )

    runtime.run()

    log_events = [event for event in trace.events if event["event"] == "log"]

    assert log_events
    assert any(event["payload"]["stage"] == "runtime_test_log" for event in log_events)


def test_task_orchestrator_writes_task_execution_summary_on_abort(tmp_path: Path):
    runtime, trace, cache, entries = _make_runtime(tmp_path, sample_count=3, role_cls=_FailingRole)

    with pytest.raises(Exception, match="boom"):
        runtime.run()

    summary = json.loads((cache.run_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["tasks"][0]["execution"]["status"] == "aborted"
    assert summary["tasks"][0]["execution"]["failed_sample_id"] == "s0"
