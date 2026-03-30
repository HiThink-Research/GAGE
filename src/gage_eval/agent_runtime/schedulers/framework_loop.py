"""Framework loop scheduler — compat wrapper shell."""

from __future__ import annotations

from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan
from gage_eval.agent_runtime.schedulers import SchedulerResult


class FrameworkLoopScheduler:
    """Compat wrapper for the legacy AgentLoop path."""

    def __init__(self, plan: CompiledRuntimePlan) -> None:
        self._plan = plan

    def run(self, session) -> SchedulerResult:
        raise NotImplementedError("FrameworkLoopScheduler is not implemented yet")
