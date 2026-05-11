from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gage_eval.assets.datasets.manager import DataManager, DataSource
from gage_eval.config.loader import load_pipeline_config_payload
from gage_eval.config.pipeline_config import (
    CustomPipelineStep,
    DatasetSpec,
    PipelineConfig,
    RoleAdapterSpec,
    TaskSpec,
)
from gage_eval.evaluation.cache import EvalCache
from gage_eval.evaluation.runtime_builder import (
    TaskOrchestratorRuntime,
    _prepare_task_entries,
    build_runtime,
)
from gage_eval.evaluation.task_batch_runtime import (
    TaskBatchExecutionContext,
    SampleLoopRuntimeEntry,
    TaskBatchRuntimeEntry,
)
from gage_eval.evaluation.task_plan import build_task_plan_specs
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.step_contracts import clear_step_contract_catalog_cache
from gage_eval.pipeline.steps.base import TaskStep
from gage_eval.pipeline.steps.report import ReportStep
from gage_eval.registry import registry
from gage_eval.reporting.recorders import InMemoryRecorder
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager


class _RecordingTaskStep(TaskStep):
    def __init__(
        self,
        *,
        step_type: str,
        adapter_id: str | None = None,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        del adapter_id, kwargs
        super().__init__(step_type)
        self.step_type = step_type
        self.params = dict(params or {})

    def execute_task(self, context, *, step, step_index: int):
        del step_index
        order = list(context.cache_store.get_metadata("task_step_order") or [])
        order.append(step.step_type)
        context.cache_store.set_metadata("task_step_order", order)
        if self.params.get("write_sample"):
            context.cache_store.write_sample(
                f"{context.task_id}-{step.step_type}",
                {
                    "sample": {"id": f"{context.task_id}-{step.step_type}"},
                    "model_output": {},
                    "judge_output": {},
                },
                namespace=context.task_id,
            )
        return {"sample_count": context.cache_store.sample_count}


class _FailingTaskStep(TaskStep):
    def __init__(self, *, step_type: str, **kwargs) -> None:
        del kwargs
        super().__init__(step_type)
        self.step_type = step_type

    def execute_task(self, context, *, step, step_index: int):
        del context, step, step_index
        raise RuntimeError("task step boom")


@pytest.fixture(scope="module", autouse=True)
def _register_task_batch_test_steps():
    registry.register(
        "pipeline_steps",
        "record_first",
        _RecordingTaskStep,
        desc="Task-batch runtime test step",
        step_kind="task",
    )
    registry.register(
        "pipeline_steps",
        "record_second",
        _RecordingTaskStep,
        desc="Task-batch runtime test step",
        step_kind="task",
    )
    registry.register(
        "pipeline_steps",
        "record_sample",
        _RecordingTaskStep,
        desc="Task-batch runtime test step that writes a sample",
        step_kind="task",
    )
    registry.register(
        "pipeline_steps",
        "fail_task",
        _FailingTaskStep,
        desc="Task-batch runtime failing test step",
        step_kind="task",
    )
    clear_step_contract_catalog_cache()
    yield
    clear_step_contract_catalog_cache()


class _RuntimeContext:
    view = None

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _RegistryStub:
    registry_view = None

    def __init__(self) -> None:
        self.materialized_dataset_ids: list[str] = []
        self.context = _RuntimeContext()

    def prepare_runtime_registry_context(self, config, *, run_id: str):
        del config, run_id
        return self.context

    def with_runtime_registry_context(self, context):
        self.context = context
        return self

    def materialize_datasets(self, config, *, trace=None):
        del trace
        self.materialized_dataset_ids = [dataset.dataset_id for dataset in config.datasets]
        return {}

    def materialize_models(self, config):
        del config
        return {}

    def materialize_backends(self, config):
        del config
        return {}

    def materialize_agent_backends(self, config, *, backends=None):
        del config, backends
        return {}

    def materialize_sandbox_profiles(self, config):
        del config
        return {}

    def materialize_mcp_clients(self, config):
        del config
        return {}

    def materialize_prompts(self, config):
        del config
        return {}

    def materialize_role_adapters(self, config, **kwargs):
        del config, kwargs
        return {}


class _EchoRole:
    adapter_id = "dut"
    role_type = "dut_model"
    resource_requirement: dict[str, Any] = {}
    sandbox_config: dict[str, Any] = {}

    def clone_for_sample(self):
        return self

    def invoke(self, payload, state=None):
        del state
        sample = payload["sample"]
        answer = f"echo-{sample.get('id')}"
        return {
            "answer": answer,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        }


def _resource_profile() -> ResourceProfile:
    return ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=1)])


def _task_batch_config(*, tasks: tuple[TaskSpec, ...] | None = None) -> PipelineConfig:
    return PipelineConfig(
        datasets=(
            DatasetSpec(
                dataset_id="harbor_ds",
                loader="harbor_local_path",
                params={"path": "/tmp/harbor-task"},
            ),
        ),
        tasks=tasks
        or (
            TaskSpec(
                task_id="tb",
                dataset_id="harbor_ds",
                execution_mode="task_batch_harness",
                steps=(
                    CustomPipelineStep(step_type="record_first"),
                    CustomPipelineStep(step_type="record_second"),
                ),
            ),
        ),
    )


@pytest.mark.fast
def test_task_batch_harness_skips_data_manager_get_and_sample_loop(tmp_path: Path, monkeypatch) -> None:
    config = _task_batch_config()
    plans = build_task_plan_specs(config)
    data_manager = DataManager()
    monkeypatch.setattr(
        data_manager,
        "get",
        lambda dataset_id: (_ for _ in ()).throw(AssertionError(f"unexpected get({dataset_id})")),
    )

    entries = _prepare_task_entries(
        task_plans=plans,
        config=config,
        data_manager=data_manager,
        datasets={},
        trace=ObservabilityTrace(recorder=InMemoryRecorder(run_id="task-batch-skip")),
        cache_store=EvalCache(base_dir=tmp_path, run_id="task-batch-skip"),
        resource_profile=_resource_profile(),
        sandbox_profiles={},
    )

    assert len(entries) == 1
    assert isinstance(entries[0], TaskBatchRuntimeEntry)
    assert not hasattr(entries[0], "sample_loop")


@pytest.mark.fast
def test_task_batch_harness_build_runtime_does_not_materialize_harbor_descriptors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    config = _task_batch_config()
    registry_stub = _RegistryStub()

    runtime = build_runtime(
        config,
        registry_stub,
        _resource_profile(),
        trace=ObservabilityTrace(recorder=InMemoryRecorder(run_id="task-batch-build")),
    )

    assert isinstance(runtime, TaskOrchestratorRuntime)
    assert registry_stub.materialized_dataset_ids == []
    runtime.shutdown()


@pytest.mark.fast
def test_sample_loop_still_calls_data_manager_and_runs_sample_loop(tmp_path: Path, monkeypatch) -> None:
    dataset = DataSource(
        dataset_id="ds",
        records=[
            {
                "id": "s0",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                "choices": [],
            }
        ],
    )
    data_manager = DataManager()
    data_manager.register_source(dataset)
    calls: list[str] = []
    original_get = data_manager.get

    def _tracked_get(dataset_id: str):
        calls.append(dataset_id)
        return original_get(dataset_id)

    monkeypatch.setattr(data_manager, "get", _tracked_get)
    config = PipelineConfig(
        datasets=(DatasetSpec(dataset_id="ds", loader="jsonl"),),
        role_adapters=(RoleAdapterSpec(adapter_id="dut", role_type="dut_model"),),
        tasks=(
            TaskSpec(
                task_id="sample_task",
                dataset_id="ds",
                steps=(CustomPipelineStep(step_type="inference", adapter_id="dut"),),
            ),
        ),
    )
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="sample-loop-still-runs"))
    cache = EvalCache(base_dir=tmp_path, run_id="sample-loop-still-runs")
    entries = _prepare_task_entries(
        task_plans=build_task_plan_specs(config),
        config=config,
        data_manager=data_manager,
        datasets={"ds": dataset},
        trace=trace,
        cache_store=cache,
        resource_profile=_resource_profile(),
        sandbox_profiles={},
    )
    role_manager = RoleManager(_resource_profile())
    role_manager.register_role_adapter("dut", _EchoRole())
    runtime = TaskOrchestratorRuntime(
        entries,
        role_manager,
        trace,
        ReportStep(auto_eval_step=None, cache_store=cache),
    )

    runtime.run()

    assert calls and calls[0] == "ds"
    assert isinstance(entries[0], SampleLoopRuntimeEntry)
    assert entries[0].sample_loop.processed_count == 1
    summary = json.loads((cache.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["tasks"][0]["execution"]["status"] == "completed"


@pytest.mark.fast
def test_task_steps_execute_in_yaml_order(tmp_path: Path) -> None:
    config = _task_batch_config()
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="task-batch-order"))
    cache = EvalCache(base_dir=tmp_path, run_id="task-batch-order")
    entries = _prepare_task_entries(
        task_plans=build_task_plan_specs(config),
        config=config,
        data_manager=DataManager(),
        datasets={},
        trace=trace,
        cache_store=cache,
        resource_profile=_resource_profile(),
        sandbox_profiles={},
    )
    runtime = TaskOrchestratorRuntime(
        entries,
        RoleManager(_resource_profile()),
        trace,
        ReportStep(auto_eval_step=None, cache_store=cache),
    )

    runtime.run()

    assert cache.get_metadata("task_step_order") == ["record_first", "record_second"]
    assert [
        event["payload"]["step_type"]
        for event in trace.events
        if event["event"] == "step_execution_completed"
    ] == ["record_first", "record_second"]


@pytest.mark.fast
def test_task_batch_request_payload_uses_top_level_external_harness_archive_with_adapter_layer(
    tmp_path: Path,
) -> None:
    config = _task_batch_config()
    task_plan = build_task_plan_specs(config)[0]
    trace = ObservabilityTrace(run_id="task-batch-raw-archive")
    cache = EvalCache(base_dir=tmp_path, run_id="task-batch-raw-archive")
    context = TaskBatchExecutionContext(
        config=config,
        task_plan=task_plan,
        task_spec=config.tasks[0],
        dataset_spec=config.datasets[0],
        role_manager=RoleManager(_resource_profile()),
        trace=trace,
        cache_store=cache,
        registry=registry,
    )

    payload = context.request_payload(adapter_id="harbor_tb2")

    archive_root = cache.run_dir / "external_harness"
    adapter_root = archive_root / "tb" / "harbor_tb2"
    assert payload["external_harness_root"] == str(archive_root)
    assert payload["external_harness_manifest_path"] == str(archive_root / "manifest.json")
    assert payload["workdir"] == str(adapter_root)
    assert payload["jobs_dir"] == str(adapter_root / "jobs")
    assert payload["job_config_path"] == str(adapter_root / "harbor_job.json")


@pytest.mark.fast
def test_task_step_failure_policy_best_effort_continues_to_next_task(tmp_path: Path) -> None:
    config = _task_batch_config(
        tasks=(
            TaskSpec(
                task_id="failing_task",
                dataset_id="harbor_ds",
                execution_mode="task_batch_harness",
                failure_policy="best_effort",
                steps=(CustomPipelineStep(step_type="fail_task"),),
            ),
            TaskSpec(
                task_id="second_task",
                dataset_id="harbor_ds",
                execution_mode="task_batch_harness",
                steps=(
                    CustomPipelineStep(
                        step_type="record_sample",
                        params={"write_sample": True},
                    ),
                ),
            ),
        )
    )
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="task-batch-best-effort"))
    cache = EvalCache(base_dir=tmp_path, run_id="task-batch-best-effort")
    runtime = TaskOrchestratorRuntime(
        _prepare_task_entries(
            task_plans=build_task_plan_specs(config),
            config=config,
            data_manager=DataManager(),
            datasets={},
            trace=trace,
            cache_store=cache,
            resource_profile=_resource_profile(),
            sandbox_profiles={},
        ),
        RoleManager(_resource_profile()),
        trace,
        ReportStep(auto_eval_step=None, cache_store=cache),
    )

    runtime.run()

    summary = json.loads((cache.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert [task["task_id"] for task in summary["tasks"]] == ["failing_task", "second_task"]
    assert summary["tasks"][0]["execution"]["status"] == "failed"
    assert summary["tasks"][0]["execution"]["failed_step"] == "fail_task"
    assert summary["tasks"][1]["execution"]["status"] == "completed"
    assert summary["tasks"][1]["sample_count"] == 1


@pytest.mark.fast
def test_existing_swebench_v2_config_defaults_to_sample_loop() -> None:
    config_path = Path("config/custom/swebench_pro/v2_local_docker_smoke.yaml")
    config = PipelineConfig.from_dict(load_pipeline_config_payload(config_path))
    plans = build_task_plan_specs(config)

    assert plans
    assert {plan.execution_mode for plan in plans} == {"sample_loop"}
    assert all(plan.runtime_policy.execution_mode == "sample_loop" for plan in plans)
