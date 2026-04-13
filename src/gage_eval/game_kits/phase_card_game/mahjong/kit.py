from __future__ import annotations

from gage_eval.game_kits.contracts import EnvSpec, GameKit
from gage_eval.game_kits.phase_card_game.mahjong.environment import MahjongArena
from gage_eval.game_kits.phase_card_game.mahjong.envs.riichi_4p import (
    Riichi4pEnvironment,
    build_riichi_4p_environment,
)
from gage_eval.game_kits.phase_card_game.mahjong.envs.riichi_4p_real import (
    Riichi4pRealEnvironment,
    build_riichi_4p_real_environment,
)
from gage_eval.game_kits.phase_card_game.mahjong.visualization import (
    VISUALIZATION_SPEC_ID,
)
from gage_eval.registry import registry


@registry.asset(
    "game_kits",
    "mahjong",
    desc="GameArena 麻将 phase-card kit",
    tags=("gamekit", "phase_card_game", "mahjong"),
)
def build_mahjong_game_kit() -> GameKit:
    return GameKit(
        kit_id="mahjong",
        family="phase_card_game",
        scheduler_binding="turn/default",
        observation_workflow="noop_observation_v1",
        visualization_spec=VISUALIZATION_SPEC_ID,
        parser="mahjong_v1",
        renderer="mahjong_replay_v1",
        input_mapper=(
            "gage_eval.game_kits.phase_card_game.mahjong.input_mapper.MahjongInputMapper"
        ),
        env_catalog=(
            EnvSpec(
                env_id="riichi_4p",
                kit_id="mahjong",
                resource_spec={"env_id": "riichi_4p", "family": "mahjong"},
                parser="mahjong_v1",
                renderer="mahjong_replay_v1",
                input_mapper=(
                    "gage_eval.game_kits.phase_card_game.mahjong.input_mapper.MahjongInputMapper"
                ),
                defaults={
                    "env_factory": build_riichi_4p_environment,
                    "replay_filename": "mahjong_riichi_4p_replay.json",
                },
            ),
            EnvSpec(
                env_id="riichi_4p_real",
                kit_id="mahjong",
                resource_spec={"env_id": "riichi_4p_real", "family": "mahjong"},
                parser="mahjong_v1",
                renderer="mahjong_replay_v1",
                input_mapper=(
                    "gage_eval.game_kits.phase_card_game.mahjong.input_mapper.MahjongInputMapper"
                ),
                defaults={
                    "env_factory": build_riichi_4p_real_environment,
                    "replay_filename": "mahjong_riichi_4p_real_replay.json",
                },
            ),
        ),
        default_env="riichi_4p",
        seat_spec={"seats": ("east", "south", "west", "north")},
        defaults={
            "replay_filename": "mahjong_riichi_4p_replay.json",
            "parser": "mahjong_v1",
            "renderer": "mahjong_replay_v1",
            "input_mapper": (
                "gage_eval.game_kits.phase_card_game.mahjong.input_mapper.MahjongInputMapper"
            ),
        },
    )


__all__ = [
    "MahjongArena",
    "Riichi4pEnvironment",
    "Riichi4pRealEnvironment",
    "build_mahjong_game_kit",
]
