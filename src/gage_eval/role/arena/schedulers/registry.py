"""Scheduler binding registry."""

from __future__ import annotations

from gage_eval.role.arena.schedulers.base import Scheduler
from gage_eval.role.arena.schedulers.real_time_tick import RealTimeTickScheduler
from gage_eval.role.arena.schedulers.record_cadence import RecordCadenceScheduler
from gage_eval.role.arena.schedulers import specs as _scheduler_specs  # noqa: F401
from gage_eval.role.arena.schedulers.turn import TurnScheduler
from gage_eval.role.arena.schedulers.specs import SchedulerBindingSpec
from gage_eval.registry import registry

_SCHEDULER_IMPLS = {
    "turn": TurnScheduler,
    "record_cadence": RecordCadenceScheduler,
    "real_time_tick": RealTimeTickScheduler,
    "placeholder://arena/schedulers/default": TurnScheduler,
}

_SCHEDULER_FAMILIES = {
    "turn": TurnScheduler,
    "agent_cycle": TurnScheduler,
    "round_robin": TurnScheduler,
    "record_cadence": RecordCadenceScheduler,
    "real_time_tick": RealTimeTickScheduler,
}


def build_scheduler_from_binding(spec: SchedulerBindingSpec) -> Scheduler:
    scheduler_impl = str(spec.scheduler_impl).strip()
    if scheduler_impl:
        scheduler_cls = _SCHEDULER_IMPLS.get(scheduler_impl)
        if scheduler_cls is None:
            raise KeyError(
                f"Unknown scheduler implementation '{spec.scheduler_impl}' "
                f"for binding '{spec.binding_id}'"
            )
    else:
        scheduler_cls = _SCHEDULER_FAMILIES.get(spec.family)
        if scheduler_cls is None:
            raise KeyError(
                f"Unknown scheduler family '{spec.family}' "
                f"for binding '{spec.binding_id}'"
            )
    return scheduler_cls(
        binding_id=spec.binding_id,
        family=spec.family,
        defaults=spec.defaults,
    )


class SchedulerRegistry:
    def __init__(self, *, registry_view=None) -> None:
        self._registry = registry_view or registry
        self.registry_view = self._registry

    def build(self, binding_id: str) -> Scheduler:
        spec = self._registry.get("scheduler_bindings", binding_id)
        return spec.build()
