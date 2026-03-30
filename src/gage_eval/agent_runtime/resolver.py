"""Resolves AgentRuntimeSpec into a CompiledRuntimePlan and scheduler."""

from __future__ import annotations

from typing import Sequence

from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan
from gage_eval.agent_runtime.schedulers import Scheduler
from gage_eval.agent_runtime.spec import AgentRuntimeSpec


class AgentRuntimeResolver:
    """Resolves runtime specs into reusable execution plans."""

    def __init__(self, runtime_specs: Sequence[AgentRuntimeSpec]) -> None:
        self._specs = {spec.agent_runtime_id: spec for spec in runtime_specs}

    def resolve(self, runtime_id: str) -> CompiledRuntimePlan:
        """Compile a runtime spec into a normalized plan."""

        spec = self._specs[runtime_id]
        return CompiledRuntimePlan(
            runtime_spec=spec,
            scheduler_type=spec.scheduler,
            benchmark_kit_id=spec.benchmark_kit_id,
            client_id=spec.client_id,
            role_adapter_id=spec.role_adapter_id,
            environment_kind=spec.resource_policy.environment_kind,
            required_surfaces=spec.client_surface_policy.required,
            optional_surfaces=spec.client_surface_policy.optional,
            sandbox_profile_id=spec.sandbox_policy.sandbox_profile_id,
            remote_mode=spec.sandbox_policy.remote_mode,
            params=spec.params,
        )

    def build_scheduler(self, plan: CompiledRuntimePlan) -> Scheduler:
        """Instantiate the correct scheduler for the plan."""

        if plan.scheduler_type == "installed_client":
            from gage_eval.agent_runtime.schedulers.installed_client import InstalledClientScheduler

            return InstalledClientScheduler(plan)
        if plan.scheduler_type == "framework_loop":
            from gage_eval.agent_runtime.schedulers.framework_loop import FrameworkLoopScheduler

            return FrameworkLoopScheduler(plan)
        if plan.scheduler_type == "acp_client":
            from gage_eval.agent_runtime.schedulers.acp_client import AcpClientScheduler

            return AcpClientScheduler(plan)
        raise ValueError(f"Unknown scheduler: {plan.scheduler_type}")
