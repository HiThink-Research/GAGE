from gage_eval.config.pipeline_config import (
    BackendSpec,
    CustomPipelineSpec,
    CustomPipelineStep,
    ModelSpec,
    PipelineConfig,
    RoleAdapterSpec,
)
from gage_eval.evaluation.cache import EvalCache
from gage_eval.evaluation.pipeline import PipelineFactory
from gage_eval.evaluation.runtime_builder import _record_config_metadata
from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.evaluation.task_planner import TaskPlanner
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager


class _RegistryStub:
    def materialize_backends(self, config):
        return {}

    def materialize_agent_runtimes(self, config):
        return {}

    def materialize_sandbox_profiles(self, config):
        return {}

    def materialize_mcp_clients(self, config):
        return {}

    def materialize_prompts(self, config):
        return {}

    def materialize_role_adapters(self, config, **kwargs):
        return {}


def _make_config() -> PipelineConfig:
    return PipelineConfig(
        metadata={"name": "metadata-consistency"},
        custom=CustomPipelineSpec(
            steps=(CustomPipelineStep(step_type="inference", adapter_id="dut"),),
        ),
        backends=(BackendSpec(backend_id="b1", type="openai", config={"model": "gpt-4o-mini"}),),
        models=(ModelSpec(model_id="m1", source="openai", params={"temperature": 0.0}),),
        role_adapters=(
            RoleAdapterSpec(
                adapter_id="dut",
                role_type="dut_model",
                backend_id="b1",
                capabilities=("chat",),
                prompt_id="prompt-main",
            ),
        ),
        summary_generators=("arena",),
    )


def test_single_and_task_runtime_metadata_share_one_contract(tmp_path) -> None:
    config = _make_config()

    pipeline_trace = ObservabilityTrace(
        recorder=InMemoryRecorder(run_id="single-runtime"),
        run_id="single-runtime",
    )
    pipeline_cache = EvalCache(base_dir=tmp_path, run_id=pipeline_trace.run_id)
    factory = PipelineFactory(_RegistryStub())
    factory.create_runtime(
        config=config,
        role_manager=RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=2)])),
        sample_loop=SampleLoop([]),
        task_planner=TaskPlanner(),
        trace=pipeline_trace,
        cache_store=pipeline_cache,
    )

    task_trace = ObservabilityTrace(
        recorder=InMemoryRecorder(run_id="task-runtime"),
        run_id="task-runtime",
    )
    task_cache = EvalCache(base_dir=tmp_path, run_id=task_trace.run_id)
    _record_config_metadata(config, task_cache, trace=task_trace)

    for key in (
        "runtime_metadata_schema_version",
        "backends",
        "agent_runtimes",
        "models",
        "role_adapters",
        "summary_generators",
    ):
        assert pipeline_cache.get_metadata(key) == task_cache.get_metadata(key)
    assert pipeline_cache.get_metadata("run_identity")["run_id"] == pipeline_trace.run_id
    assert task_cache.get_metadata("run_identity")["run_id"] == task_trace.run_id
