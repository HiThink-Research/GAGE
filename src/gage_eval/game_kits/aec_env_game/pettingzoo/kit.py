"""PettingZoo GameKit registry entry."""

from __future__ import annotations

from gage_eval.game_kits.aec_env_game.pettingzoo.envs.space_invaders import (
    build_space_invaders_environment,
)
from gage_eval.game_kits.contracts import EnvSpec, GameKit
from gage_eval.game_kits.aec_env_game.pettingzoo.visualization import (
    VISUALIZATION_SPEC_ID,
)
from gage_eval.registry import registry


@registry.asset(
    "game_kits",
    "pettingzoo",
    desc="GameArena PettingZoo kit",
    tags=("gamekit", "aec_env_game", "pettingzoo"),
)
def build_pettingzoo_game_kit() -> GameKit:
    return GameKit(
        kit_id="pettingzoo",
        family="aec_env_game",
        scheduler_binding="agent_cycle/default",
        observation_workflow="noop_observation_v1",
        visualization_spec=VISUALIZATION_SPEC_ID,
        parser=(
            "gage_eval.game_kits.aec_env_game.pettingzoo.action_codec.DiscreteActionParser"
        ),
        input_mapper=(
            "gage_eval.game_kits.aec_env_game.pettingzoo.input_mapper.PettingZooDiscreteInputMapper"
        ),
        env_catalog=(
            EnvSpec(
                env_id="space_invaders",
                kit_id="pettingzoo",
                resource_spec={
                    "env_id": "space_invaders",
                    "family": "pettingzoo",
                },
                parser=(
                    "gage_eval.game_kits.aec_env_game.pettingzoo.action_codec.DiscreteActionParser"
                ),
                input_mapper=(
                    "gage_eval.game_kits.aec_env_game.pettingzoo.input_mapper.PettingZooDiscreteInputMapper"
                ),
                defaults={
                    "env_factory": build_space_invaders_environment,
                    "env_id": "pettingzoo.atari.space_invaders_v2",
                    "max_cycles": 4,
                    "seed": 7,
                    "use_action_meanings": True,
                    "include_raw_obs": False,
                    "illegal_policy": {"retry": 0, "on_fail": "loss"},
                },
            ),
        ),
        default_env="space_invaders",
        seat_spec={"seats": ("pilot_0", "pilot_1")},
        defaults={
            "env_id": "pettingzoo.atari.space_invaders_v2",
            "max_cycles": 4,
            "seed": 7,
            "use_action_meanings": True,
            "include_raw_obs": False,
            "illegal_policy": {"retry": 0, "on_fail": "loss"},
            "parser": (
                "gage_eval.game_kits.aec_env_game.pettingzoo.action_codec.DiscreteActionParser"
            ),
            "input_mapper": (
                "gage_eval.game_kits.aec_env_game.pettingzoo.input_mapper.PettingZooDiscreteInputMapper"
            ),
        },
    )


__all__ = ["build_pettingzoo_game_kit"]
