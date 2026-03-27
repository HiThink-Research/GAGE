"""Support workflow runner."""

from __future__ import annotations

from dataclasses import dataclass, field

from gage_eval.role.arena.support.context import SupportContext
from gage_eval.role.arena.support.hooks import SupportHook, SupportUnit


@dataclass
class GameSupportWorkflow:
    workflow_id: str
    units_by_hook: dict[SupportHook, list[SupportUnit]] = field(default_factory=dict)

    def run(self, hook: SupportHook, context: SupportContext) -> SupportContext:
        current = context
        for unit in self.units_by_hook.get(hook, []):
            current = unit.invoke(current)
        return current
