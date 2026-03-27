from __future__ import annotations

from gage_eval.game_kits.contracts import EnvSpec, GameKit
from gage_eval.game_kits.real_time_game.vizdoom.envs.duel_map01 import (
    build_duel_map01_environment,
)
from gage_eval.registry import registry


@registry.asset(
    "game_kits",
    "vizdoom",
    desc="GameArena ViZDoom realtime kit",
    tags=("gamekit", "real_time_game", "vizdoom"),
)
def build_vizdoom_game_kit() -> GameKit:
    return GameKit(
        kit_id="vizdoom",
        family="real_time_game",
        scheduler_binding="real_time_tick/default",
        observation_workflow="noop_observation_v1",
        env_catalog=(
            EnvSpec(
                env_id="duel_map01",
                kit_id="vizdoom",
                resource_spec={"env_id": "duel_map01", "family": "vizdoom"},
                defaults={
                    "env_factory": build_duel_map01_environment,
                    "backend_mode": "real",
                    "stub_max_rounds": 3,
                    "max_steps": 12,
                    "show_pov": False,
                    "show_automap": False,
                    "allow_partial_actions": False,
                    "replay_in_env": True,
                },
            ),
        ),
        default_env="duel_map01",
        seat_spec={"seats": ("p0", "p1")},
        defaults={
            "backend_mode": "real",
            "stub_max_rounds": 3,
            "max_steps": 12,
            "show_pov": False,
            "show_automap": False,
            "allow_partial_actions": False,
            "replay_in_env": True,
        },
    )
