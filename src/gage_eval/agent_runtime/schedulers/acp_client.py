"""ACP client scheduler — shell only, not yet implemented."""

from __future__ import annotations

from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan
from gage_eval.agent_runtime.schedulers import SchedulerResult


class AcpClientScheduler:
    """Placeholder for ACP-based client execution."""

    def __init__(self, plan: CompiledRuntimePlan) -> None:
        self._plan = plan

    def run(self, session) -> SchedulerResult:
        raise NotImplementedError("AcpClientScheduler is not implemented yet")
