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

    plan_id: str
    runtime_spec: AgentRuntimeSpec
    scheduler_handle: Any
    workflow_bundle: SchedulerWorkflowBundle
    kit_runtime_ref: Any
    judge_binding: JudgeBinding
    resource_plan: dict[str, Any]
    artifact_policy: dict[str, Any]
    cache_key: str
    benchmark_config: dict[str, Any] = field(default_factory=dict)
    compile_diagnostics: list[dict[str, Any]] = field(default_factory=list)
