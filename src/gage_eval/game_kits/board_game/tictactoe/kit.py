from __future__ import annotations

from gage_eval.game_kits.contracts import EnvSpec, GameKit
from gage_eval.game_kits.board_game.tictactoe.envs.tictactoe_standard import (
    TicTacToeStandardEnvironment,
    build_tictactoe_standard_environment,
)
from gage_eval.game_kits.board_game.tictactoe.visualization import (
    VISUALIZATION_SPEC_ID,
)
from gage_eval.registry import registry


@registry.asset(
    "game_kits",
    "tictactoe",
    desc="GameArena Tic-Tac-Toe kit",
    tags=("gamekit", "board_game", "tictactoe"),
)
def build_tictactoe_game_kit() -> GameKit:
    return GameKit(
        kit_id="tictactoe",
        family="board_game",
        scheduler_binding="turn/default",
        observation_workflow="noop_observation_v1",
        game_display=VISUALIZATION_SPEC_ID,
        replay_viewer=VISUALIZATION_SPEC_ID,
        parser="grid_parser_v1",
        visualization_spec=VISUALIZATION_SPEC_ID,
        env_catalog=(
            EnvSpec(
                env_id="tictactoe_standard",
                kit_id="tictactoe",
                resource_spec={"env_id": "tictactoe_standard", "family": "tictactoe"},
                defaults={
                    "env_factory": build_tictactoe_standard_environment,
                    "board_size": 3,
                    "coord_scheme": "ROW_COL",
                },
            ),
        ),
        default_env="tictactoe_standard",
        seat_spec={"seats": ("x", "o")},
        defaults={
            "board_size": 3,
            "coord_scheme": "ROW_COL",
        },
    )


__all__ = ["TicTacToeStandardEnvironment", "build_tictactoe_game_kit"]
