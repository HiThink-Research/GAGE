from __future__ import annotations

import pytest

from gage_eval.role.arena.schedulers.specs import SchedulerBindingSpec
from gage_eval.role.arena.schedulers.registry import SchedulerRegistry
from gage_eval.registry import registry


def test_scheduler_registry_builds_record_cadence_binding() -> None:
    scheduler = SchedulerRegistry().build("record_cadence/default")
    assert scheduler.family == "record_cadence"


def test_scheduler_registry_builds_realtime_bindings() -> None:
    vizdoom_scheduler = SchedulerRegistry().build("real_time_tick/default")
    retro_scheduler = SchedulerRegistry().build("real_time_tick/default")

    assert vizdoom_scheduler.family == "real_time_tick"
    assert vizdoom_scheduler.binding_id == "real_time_tick/default"
    assert retro_scheduler.family == "real_time_tick"
    assert retro_scheduler.binding_id == "real_time_tick/default"


def test_scheduler_registry_builds_pettingzoo_bindings() -> None:
    cycle_scheduler = SchedulerRegistry().build("agent_cycle/default")
    record_scheduler = SchedulerRegistry().build("record_cadence/default")

    assert cycle_scheduler.family == "agent_cycle"
    assert cycle_scheduler.binding_id == "agent_cycle/default"
    assert record_scheduler.family == "record_cadence"
    assert record_scheduler.binding_id == "record_cadence/default"


def test_scheduler_registry_builds_default_binding_with_alias_family() -> None:
    scheduler = SchedulerRegistry().build("turn/default")
    assert scheduler.family == "turn"
    assert scheduler.binding_id == "turn/default"


def test_scheduler_registry_propagates_binding_defaults() -> None:
    scheduler = SchedulerRegistry().build("turn/default")
    assert scheduler.defaults == {"max_ticks": 256}


def test_scheduler_registry_raises_for_invalid_scheduler_impl() -> None:
    invalid_binding = SchedulerBindingSpec(
        binding_id="arena_v2/invalid_impl",
        family="turn",
        scheduler_impl="unknown://scheduler/impl",
        defaults={"max_ticks": 64},
    )
    clone = registry.clone()
    with registry.route_to(clone):
        registry.register(
            "scheduler_bindings",
            invalid_binding.binding_id,
            invalid_binding,
            desc="Invalid scheduler binding for tests",
        )
        with pytest.raises(KeyError, match="Unknown scheduler implementation"):
            SchedulerRegistry().build(invalid_binding.binding_id)
