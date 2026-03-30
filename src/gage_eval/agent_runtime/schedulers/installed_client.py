"""Installed client scheduler — shell only until the client path is implemented."""

from __future__ import annotations

from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan
from gage_eval.agent_runtime.schedulers import SchedulerResult


class InstalledClientScheduler:
    """Runs an installed CLI client inside the selected environment."""

    def __init__(self, plan: CompiledRuntimePlan) -> None:
        self._plan = plan

    def run(self, session) -> SchedulerResult:
        raise NotImplementedError("InstalledClientScheduler is not implemented yet")
