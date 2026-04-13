from __future__ import annotations

from gage_eval.game_kits.contracts import EnvSpec, GameKit
from gage_eval.game_kits.board_game.gomoku.envs.gomoku_standard import (
    GomokuStandardEnvironment,
    build_gomoku_standard_environment,
)
from gage_eval.game_kits.board_game.gomoku.visualization import (
    VISUALIZATION_SPEC_ID,
)
from gage_eval.registry import registry


@registry.asset(
    "game_kits",
    "gomoku",
    desc="GameArena Gomoku kit",
    tags=("gamekit", "board_game", "gomoku"),
)
def build_gomoku_game_kit() -> GameKit:
    return GameKit(
        kit_id="gomoku",
        family="board_game",
        scheduler_binding="turn/default",
        observation_workflow="noop_observation_v1",
        game_display=VISUALIZATION_SPEC_ID,
        replay_viewer=VISUALIZATION_SPEC_ID,
        parser="gomoku_v1",
        visualization_spec=VISUALIZATION_SPEC_ID,
        env_catalog=(
            EnvSpec(
                env_id="gomoku_standard",
                kit_id="gomoku",
                resource_spec={"env_id": "gomoku_standard", "family": "gomoku"},
                defaults={
                    "env_factory": build_gomoku_standard_environment,
                    "board_size": 15,
                    "win_len": 5,
                    "coord_scheme": "A1",
                    "rule_profile": "freestyle",
                    "win_directions": (
                        "horizontal",
                        "vertical",
                        "diagonal",
                        "anti_diagonal",
                    ),
                },
            ),
        ),
        default_env="gomoku_standard",
        seat_spec={"seats": ("black", "white")},
        defaults={
            "board_size": 15,
            "win_len": 5,
            "coord_scheme": "A1",
            "rule_profile": "freestyle",
        },
    )


__all__ = ["GomokuStandardEnvironment", "build_gomoku_game_kit"]
