from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from gage_eval.agent_runtime.spec import AgentRuntimeSpec
from gage_eval.agent_runtime.verifier.binding import JudgeBinding


class RuntimeCompileError(ValueError):
    """Signals that cold-path runtime compilation failed with diagnostics."""

    def __init__(self, message: str, *, diagnostics: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = list(diagnostics or [])


@dataclass(frozen=True)
class SchedulerWorkflowBundle:
    """Captures the scheduler-local workflow projection functions."""

    bundle_id: str
    benchmark_kit_id: str
    scheduler_type: str
    prepare_inputs: Callable[..., dict[str, Any]] | None = None
    prepare_environment: Callable[..., dict[str, Any]] | None = None
    capture_environment_artifacts: Callable[..., dict[str, str]] | None = None
    finalize_result: Callable[..., dict[str, Any]] | None = None
    build_loop_inputs: Callable[..., dict[str, Any]] | None = None
    inject_prompt_context: Callable[..., dict[str, Any]] | None = None
    inject_tool_schemas: Callable[..., list[dict[str, Any]]] | None = None
    finalize_loop_result: Callable[..., dict[str, Any]] | None = None
    failure_normalizer: Callable[..., dict[str, Any]] | None = None


@dataclass(frozen=True)
class CompiledRuntimePlan:
    """Captures the cold-path runtime compilation output."""

    run_id: str
    dut_id: str
    agent_id: str
    env_id: str
    benchmark_id: str
    trial_policy: dict[str, Any]
    kit_id: str
    kit_entry: Any
    kit_config: dict[str, Any]
    agent_config: dict[str, Any]
    scheduler_type: str
    scheduler_config: dict[str, Any]
    environment_provider: str
    environment_profile_id: str
    environment_profile: dict[str, Any]
    lifecycle: str
    provider_config: dict[str, Any]
    startup_env: dict[str, Any]
    resources: dict[str, Any]
    verifier_environment_policy: str
    verifier_environment_profile_id: str | None
    workflow_bundle: SchedulerWorkflowBundle
    tool_registry: Any
    tool_provider_adapter: Any
    verifier_adapter: Any
    artifact_sink: Any
    plan_id: str | None = None
    runtime_spec: AgentRuntimeSpec | None = None
    scheduler_handle: Any = None
    kit_runtime_ref: Any = None
    judge_binding: JudgeBinding | None = None
    resource_plan: dict[str, Any] = field(default_factory=dict)
    artifact_policy: dict[str, Any] = field(default_factory=dict)
    cache_key: str = ""
    compile_diagnostics: list[dict[str, Any]] = field(default_factory=list)
