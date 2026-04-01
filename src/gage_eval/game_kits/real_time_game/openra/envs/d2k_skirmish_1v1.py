"""OpenRA Dune 2000 1v1 skirmish smoke environment."""

from __future__ import annotations

from gage_eval.game_kits.real_time_game.openra.environment import OpenRAArenaEnvironment


def build_d2k_skirmish_1v1_environment(
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


__all__ = ["OpenRAArenaEnvironment", "build_d2k_skirmish_1v1_environment"]
