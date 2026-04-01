"""OpenRA Red Alert skirmish smoke environment."""

from __future__ import annotations

from gage_eval.game_kits.real_time_game.openra.environment import OpenRAArenaEnvironment


def build_ra_map01_environment(
    *,
    sample,
    resolved,
    resources,
    player_specs,
    invocation_context=None,
) -> OpenRAArenaEnvironment:
    return OpenRAArenaEnvironment.from_runtime(
        sample=sample,
        resolved=resolved,
        resources=resources,
        player_specs=player_specs,
        invocation_context=invocation_context,
    )


__all__ = ["build_ra_map01_environment", "OpenRAArenaEnvironment"]
