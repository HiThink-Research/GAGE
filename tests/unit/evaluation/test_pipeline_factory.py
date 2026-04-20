from __future__ import annotations

from gage_eval.config.pipeline_config import (
    BackendSpec,
    CustomPipelineSpec,
    CustomPipelineStep,
    PipelineConfig,
)
from gage_eval.evaluation.pipeline import PipelineFactory
from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.evaluation.task_planner import TaskPlanner
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager


class _RegistryStub:
    @property
    def registry_view(self):
        return None

    def materialize_backends(self, config):
        _ = config
        return {"model_backend": object()}

    def materialize_agent_runtimes(self, config):
        _ = config
        return {}

    def materialize_sandbox_profiles(self, config):
        _ = config
        return {}

    def materialize_mcp_clients(self, config):
        _ = config
        return {}

    def materialize_prompts(self, config):
        _ = config
        return {}

    def materialize_role_adapters(self, config, **kwargs):
        _ = config, kwargs
        return {}


def test_pipeline_factory_registers_global_backends_on_role_manager() -> None:
    config = PipelineConfig(
        metadata={"name": "pipeline-factory-backends"},
        custom=CustomPipelineSpec(
            steps=(CustomPipelineStep(step_type="arena", adapter_id="arena"),),
        ),
        backends=(
            BackendSpec(
                backend_id="model_backend",
                type="litellm",
                config={"model": "qwen/qwen3.5-9b"},
            ),
        ),
    )
    role_manager = RoleManager(
        ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=2)])
    )

    factory = PipelineFactory(_RegistryStub())
    factory.create_runtime(
        config=config,
        role_manager=role_manager,
        sample_loop=SampleLoop([]),
        task_planner=TaskPlanner(),
        trace=ObservabilityTrace(run_id="pipeline-factory-backends"),
    )

    assert role_manager.get_backend("model_backend") is not None
