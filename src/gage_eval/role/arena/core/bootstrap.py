from __future__ import annotations

import inspect

from gage_eval.role.arena.core.arena_core import GameArenaCore
from gage_eval.role.arena.output.writer import ArenaOutputWriter
from gage_eval.role.arena.player_drivers.registry import PlayerDriverRegistry
from gage_eval.role.arena.resources.control import ArenaResourceControl
from gage_eval.role.arena.schedulers.registry import SchedulerRegistry
from gage_eval.role.arena.support.registry import SupportWorkflowRegistry
from gage_eval.game_kits.registry import GameKitRegistry, ObservationWorkflowRegistry
from gage_eval.game_kits.runtime_binding import RuntimeBindingResolver


def build_gamearena_core(
    *,
    registry_view=None,
    game_kits: GameKitRegistry | None = None,
    schedulers: SchedulerRegistry | None = None,
    observation_workflows: ObservationWorkflowRegistry | None = None,
    support_workflows: SupportWorkflowRegistry | None = None,
    player_drivers: PlayerDriverRegistry | None = None,
    resource_control: ArenaResourceControl | None = None,
    output_writer: ArenaOutputWriter | None = None,
) -> GameArenaCore:
    game_kits = game_kits or GameKitRegistry(registry_view=registry_view)
    registry_source = getattr(game_kits, "registry_view", registry_view)
    schedulers = schedulers or SchedulerRegistry(registry_view=registry_source)
    observation_workflows = observation_workflows or ObservationWorkflowRegistry(
        registry_view=registry_source
    )
    support_workflows = support_workflows or SupportWorkflowRegistry(
        registry_view=registry_source
    )
    player_drivers = player_drivers or PlayerDriverRegistry(
        registry_view=registry_source
    )
    resolver_kwargs = {
        "game_kits": game_kits,
        "schedulers": schedulers,
        "observation_workflows": observation_workflows,
        "support_workflows": support_workflows,
        "player_drivers": player_drivers,
    }
    resolver = _build_runtime_binding_resolver(**resolver_kwargs)
    resource_control = resource_control or ArenaResourceControl()
    output_writer = output_writer or ArenaOutputWriter()
    return GameArenaCore(
        resolver=resolver,
        resource_control=resource_control,
        output_writer=output_writer,
    )


__all__ = ["build_gamearena_core"]


def _build_runtime_binding_resolver(**kwargs):
    try:
        signature = inspect.signature(RuntimeBindingResolver)
    except (TypeError, ValueError):
        return RuntimeBindingResolver(**kwargs)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return RuntimeBindingResolver(**kwargs)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return RuntimeBindingResolver(**supported)
