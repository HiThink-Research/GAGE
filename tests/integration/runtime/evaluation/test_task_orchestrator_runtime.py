from pathlib import Path

from gage_eval.config.pipeline_config import (
    CustomPipelineStep,
    DatasetSpec,
    PipelineConfig,
    RoleAdapterSpec,
    TaskSpec,
)
from gage_eval.evaluation.cache import EvalCache
from gage_eval.evaluation.runtime_builder import TaskOrchestratorRuntime, _prepare_task_entries, _record_config_metadata
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager
from gage_eval.assets.datasets.manager import DataManager, DataSource
from gage_eval.pipeline.steps.report import ReportStep
from gage_eval.metrics import MetricRegistry
from gage_eval.evaluation.task_plan import build_task_plan_specs


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


def _make_runtime(tmp_path: Path, sample_count: int = 4):
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
    cache = EvalCache(base_dir=tmp_path, run_id="runtime-test")
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="runtime-trace"))
    report_step = ReportStep(auto_eval_step=None, cache_store=cache)
    entries = _prepare_task_entries(
        task_plans=plans,
        config=config,
        data_manager=dm,
        datasets={"ds": dataset},
        trace=trace,
        cache_store=cache,
        resource_profile=ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=2)]),
    )

    rm = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=2)]))
    rm.register_role_adapter("dut", _EchoRole("dut", "dut_model"))

    runtime = TaskOrchestratorRuntime(entries, rm, trace, report_step)
    _record_config_metadata(config, cache)
    return runtime, trace, cache, entries


def test_task_orchestrator_runs_tasks(tmp_path: Path):
    runtime, trace, cache, entries = _make_runtime(tmp_path)

    runtime.run()

    task_starts = [e for e in trace.events if e["event"] == "task_start"]
    task_ends = [e for e in trace.events if e["event"] == "task_end"]
    assert len(task_starts) == 2
    assert len(task_ends) == 2
    assert all(entry.sample_loop.processed_count > 0 for entry in entries)
    summary = cache.run_dir / "summary.json"
    assert summary.exists()


def test_task_orchestrator_records_timings(tmp_path: Path):
    runtime, trace, cache, entries = _make_runtime(tmp_path, sample_count=2)
    runtime.run()

    timings = cache._timings
    assert "inference_s" in timings and timings["inference_s"] >= 0.0
