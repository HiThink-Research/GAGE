from __future__ import annotations

from typing import Any

__all__ = [
    "AgentRuntimeSession",
    "AgentRuntimeSpec",
    "CompiledRuntimeExecutor",
    "build_compiled_runtime_executor",
    "compile_agent_runtime_plan",
    "CompiledRuntimePlan",
    "RuntimeCompileError",
    "FailureEnvelope",
    "FailureEnvelopeError",
    "JudgeBinding",
    "ResourceLease",
    "RuntimeJudgeOutcome",
    "SchedulerResult",
    "SchedulerWorkflowBundle",
    "resolve_agent_runtime_spec",
]


def __getattr__(name: str) -> Any:
    """Lazily expose agent runtime symbols."""

    if name in {"CompiledRuntimePlan", "RuntimeCompileError", "SchedulerWorkflowBundle"}:
        from gage_eval.agent_runtime.compiled_plan import (
            CompiledRuntimePlan,
            RuntimeCompileError,
            SchedulerWorkflowBundle,
        )

        return {
            "CompiledRuntimePlan": CompiledRuntimePlan,
            "RuntimeCompileError": RuntimeCompileError,
            "SchedulerWorkflowBundle": SchedulerWorkflowBundle,
        }[name]
    if name in {"FailureEnvelope", "FailureEnvelopeError"}:
        from gage_eval.agent_runtime.contracts.failure import FailureEnvelope, FailureEnvelopeError

        return {
            "FailureEnvelope": FailureEnvelope,
            "FailureEnvelopeError": FailureEnvelopeError,
        }[name]
    if name == "SchedulerResult":
        from gage_eval.agent_runtime.contracts.scheduler import SchedulerResult

        return SchedulerResult
    if name == "CompiledRuntimeExecutor":
        from gage_eval.agent_runtime.executor import CompiledRuntimeExecutor

        return CompiledRuntimeExecutor
    if name in {
        "build_compiled_runtime_executor",
        "compile_agent_runtime_plan",
        "resolve_agent_runtime_spec",
    }:
        from gage_eval.agent_runtime.resolver import (
            build_compiled_runtime_executor,
            compile_agent_runtime_plan,
            resolve_agent_runtime_spec,
        )

        return {
            "build_compiled_runtime_executor": build_compiled_runtime_executor,
            "compile_agent_runtime_plan": compile_agent_runtime_plan,
            "resolve_agent_runtime_spec": resolve_agent_runtime_spec,
        }[name]
    if name == "ResourceLease":
        from gage_eval.agent_runtime.resources.contracts import ResourceLease

        return ResourceLease
    if name == "AgentRuntimeSession":
        from gage_eval.agent_runtime.session import AgentRuntimeSession

        return AgentRuntimeSession
    if name == "AgentRuntimeSpec":
        from gage_eval.agent_runtime.spec import AgentRuntimeSpec

        return AgentRuntimeSpec
    if name == "JudgeBinding":
        from gage_eval.agent_runtime.verifier.binding import JudgeBinding

        return JudgeBinding
    if name == "RuntimeJudgeOutcome":
        from gage_eval.agent_runtime.verifier.contracts import RuntimeJudgeOutcome

        return RuntimeJudgeOutcome
    raise AttributeError(name)
