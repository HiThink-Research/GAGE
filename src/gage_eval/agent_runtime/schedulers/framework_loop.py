"""Framework loop scheduler — thin compat wrapper around AgentLoop."""

from __future__ import annotations

from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan
from gage_eval.agent_runtime.schedulers import SchedulerResult


class FrameworkLoopScheduler:
    """Compat wrapper for the legacy AgentLoop path."""

    def __init__(self, plan: CompiledRuntimePlan) -> None:
        self._plan = plan

    def run(self, session) -> SchedulerResult:
        from gage_eval.role.agent.loop import AgentLoop

        backend = session.metadata.get("agent_backend")
        tool_router = session.metadata.get("tool_router")
        if backend is None or tool_router is None:
            return SchedulerResult(
                status="error",
                raw_output={"error": "framework_loop_context_missing"},
            )
        loop = AgentLoop(
            backend=backend,
            tool_router=tool_router,
            max_turns=int(session.metadata.get("max_turns", 8)),
        )
        result = loop.run(
            messages=list(session.sample.get("messages") or []),
            tools=list(session.sample.get("tools") or []),
            tool_choice=session.sample.get("tool_choice"),
            metadata=session.sample.get("metadata") or {},
            sample=session.sample,
        )
        return SchedulerResult(
            status="success",
            answer=result.get("answer"),
            raw_output=dict(result),
        )
