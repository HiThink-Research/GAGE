"""Scheduler binding specs for GameArena."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gage_eval.registry import registry

if TYPE_CHECKING:
    from gage_eval.role.arena.schedulers.base import Scheduler


@dataclass(frozen=True)
class SchedulerBindingSpec:
    binding_id: str
    family: str
    scheduler_impl: str
    defaults: dict[str, object] = field(default_factory=dict)

    def build(self) -> "Scheduler":
        from gage_eval.role.arena.schedulers.registry import build_scheduler_from_binding

        return build_scheduler_from_binding(self)


DEFAULT_SCHEDULER_BINDING = SchedulerBindingSpec(
    binding_id="turn/default",
    family="turn",
    scheduler_impl="turn",
    defaults={"max_ticks": 256},
)

AGENT_CYCLE_DEFAULT_BINDING = SchedulerBindingSpec(
    binding_id="agent_cycle/default",
    family="agent_cycle",
    scheduler_impl="turn",
    defaults={"max_ticks": 256},
)

RECORD_CADENCE_DEFAULT_BINDING = SchedulerBindingSpec(
    binding_id="record_cadence/default",
    family="record_cadence",
    scheduler_impl="record_cadence",
    defaults={"max_ticks": 256},
)

REAL_TIME_TICK_DEFAULT_BINDING = SchedulerBindingSpec(
    binding_id="real_time_tick/default",
    family="real_time_tick",
    scheduler_impl="real_time_tick",
    defaults={"max_ticks": 256},
)

registry.register(
    "scheduler_bindings",
    "turn/default",
    DEFAULT_SCHEDULER_BINDING,
    desc="Default scheduler binding for GameArena",
)

registry.register(
    "scheduler_bindings",
    "agent_cycle/default",
    AGENT_CYCLE_DEFAULT_BINDING,
    desc="Default agent-cycle scheduler binding for GameArena",
)
registry.register(
    "scheduler_bindings",
    "record_cadence/default",
    RECORD_CADENCE_DEFAULT_BINDING,
    desc="Default record-cadence scheduler binding for GameArena",
)
registry.register(
    "scheduler_bindings",
    "real_time_tick/default",
    REAL_TIME_TICK_DEFAULT_BINDING,
    desc="Default realtime-tick scheduler binding for GameArena",
)


def register_runtime_assets(*, registry_target=None) -> None:
    target = registry_target or registry
    target.register(
        "scheduler_bindings",
        "turn/default",
        DEFAULT_SCHEDULER_BINDING,
        desc="Default scheduler binding for GameArena",
    )
    target.register(
        "scheduler_bindings",
        "agent_cycle/default",
        AGENT_CYCLE_DEFAULT_BINDING,
        desc="Default agent-cycle scheduler binding for GameArena",
    )
    target.register(
        "scheduler_bindings",
        "record_cadence/default",
        RECORD_CADENCE_DEFAULT_BINDING,
        desc="Default record-cadence scheduler binding for GameArena",
    )
    target.register(
        "scheduler_bindings",
        "real_time_tick/default",
        REAL_TIME_TICK_DEFAULT_BINDING,
        desc="Default realtime-tick scheduler binding for GameArena",
    )


__all__ = [
    "SchedulerBindingSpec",
    "DEFAULT_SCHEDULER_BINDING",
    "AGENT_CYCLE_DEFAULT_BINDING",
    "RECORD_CADENCE_DEFAULT_BINDING",
    "REAL_TIME_TICK_DEFAULT_BINDING",
    "register_runtime_assets",
]
