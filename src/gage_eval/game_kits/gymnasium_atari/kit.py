"""Gymnasium Atari GameKit registry entry."""

from __future__ import annotations

from gage_eval.game_kits.aec_env_game.pettingzoo.visualization import (
    VISUALIZATION_SPEC_ID,
)
from gage_eval.game_kits.contracts import EnvSpec, GameKit
from gage_eval.game_kits.gymnasium_atari.envs.space_invaders import (
    build_space_invaders_environment,
)
from gage_eval.registry import registry

_DISCRETE_PARSER = (
    "gage_eval.game_kits.aec_env_game.pettingzoo.action_codec.DiscreteActionParser"
)
_DISCRETE_INPUT_MAPPER = (
    "gage_eval.game_kits.aec_env_game.pettingzoo.input_mapper.PettingZooDiscreteInputMapper"
)


@registry.asset(
    "game_kits",
    "gymnasium_atari",
    desc="GameArena Gymnasium Atari kit",
    tags=("gamekit", "gymnasium", "atari"),
)
def build_gymnasium_atari_game_kit() -> GameKit:
    """Builds the single-player Gymnasium Atari GameKit."""

    defaults = {
        "env_factory": build_space_invaders_environment,
        "backend_mode": "real",
        "env_id": "ALE/SpaceInvaders-v5",
        "max_cycles": 7200,
        "seed": 7,
        "use_action_meanings": True,
        "include_raw_obs": False,
        "illegal_policy": {"retry": 0, "on_fail": "loss"},
        "action_schema": {
            "hold_ticks_min": 1,
            "hold_ticks_max": 4,
            "hold_ticks_default": 1,
        },
    }
    return GameKit(
        kit_id="gymnasium_atari",
        family="gymnasium_atari",
        scheduler_binding="real_time_tick/default",
        observation_workflow="noop_observation_v1",
        visualization_spec=VISUALIZATION_SPEC_ID,
        parser=_DISCRETE_PARSER,
        input_mapper=_DISCRETE_INPUT_MAPPER,
        env_catalog=(
            EnvSpec(
                env_id="space_invaders",
                kit_id="gymnasium_atari",
                resource_spec={
                    "env_id": "space_invaders",
                    "family": "gymnasium_atari",
                },
                parser=_DISCRETE_PARSER,
                input_mapper=_DISCRETE_INPUT_MAPPER,
                defaults=dict(defaults),
            ),
        ),
        default_env="space_invaders",
        seat_spec={"seats": ("pilot_0",)},
        defaults={
            **defaults,
            "parser": _DISCRETE_PARSER,
            "input_mapper": _DISCRETE_INPUT_MAPPER,
        },
    )


__all__ = ["build_gymnasium_atari_game_kit"]
