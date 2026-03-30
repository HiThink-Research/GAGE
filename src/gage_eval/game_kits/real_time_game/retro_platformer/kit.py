from __future__ import annotations

from gage_eval.game_kits.contracts import EnvSpec, GameKit
from gage_eval.game_kits.real_time_game.retro_platformer.envs.retro_mario import (
    RetroMarioEnvironment,
    build_retro_mario_environment,
)
from gage_eval.game_kits.real_time_game.retro_platformer.visualization import (
    VISUALIZATION_SPEC_ID,
)
from gage_eval.registry import registry


@registry.asset(
    "game_kits",
    "retro_platformer",
    desc="GameArena retro platformer realtime kit",
    tags=("gamekit", "real_time_game", "retro"),
)
def build_retro_platformer_game_kit() -> GameKit:
    return GameKit(
        kit_id="retro_platformer",
        family="real_time_game",
        scheduler_binding="real_time_tick/default",
        observation_workflow="noop_observation_v1",
        visualization_spec=VISUALIZATION_SPEC_ID,
        parser=(
            "gage_eval.game_kits.real_time_game.retro_platformer.parser.RetroActionParser"
        ),
        input_mapper=(
            "gage_eval.game_kits.real_time_game.retro_platformer.input_mapper.RetroInputMapper"
        ),
        env_catalog=(
            EnvSpec(
                env_id="retro_mario",
                kit_id="retro_platformer",
                resource_spec={"env_id": "retro_mario", "family": "retro"},
                parser=(
                    "gage_eval.game_kits.real_time_game.retro_platformer.parser.RetroActionParser"
                ),
                input_mapper=(
                    "gage_eval.game_kits.real_time_game.retro_platformer.input_mapper.RetroInputMapper"
                ),
                defaults={
                    "env_factory": build_retro_mario_environment,
                    "backend_mode": "real",
                    "stub_max_ticks": 4,
                    "game": "SuperMarioBros3-Nes-v0",
                    "display_mode": "headless",
                    "obs_image": True,
                    "record_bk2": False,
                    "legal_moves": (
                        "noop",
                        "right",
                        "right_run",
                        "right_jump",
                        "right_run_jump",
                        "jump",
                    ),
                    "action_schema": {
                        "hold_ticks_min": 1,
                        "hold_ticks_max": 12,
                        "hold_ticks_default": 6,
                    },
                    "info_feeder": {"impl": "info_last_v1"},
                },
            ),
        ),
        default_env="retro_mario",
        seat_spec={"seats": ("player_0",)},
        defaults={
            "backend_mode": "real",
            "stub_max_ticks": 4,
            "game": "SuperMarioBros3-Nes-v0",
            "display_mode": "headless",
            "obs_image": True,
            "record_bk2": False,
            "legal_moves": (
                "noop",
                "right",
                "right_run",
                "right_jump",
                "right_run_jump",
                "jump",
            ),
            "action_schema": {
                "hold_ticks_min": 1,
                "hold_ticks_max": 12,
                "hold_ticks_default": 6,
            },
            "info_feeder": {"impl": "info_last_v1"},
            "parser": (
                "gage_eval.game_kits.real_time_game.retro_platformer.parser.RetroActionParser"
            ),
            "input_mapper": (
                "gage_eval.game_kits.real_time_game.retro_platformer.input_mapper.RetroInputMapper"
            ),
        },
    )


__all__ = ["RetroMarioEnvironment", "build_retro_platformer_game_kit"]
