from __future__ import annotations

from gage_eval.role.arena.types import ArenaAction
from gage_eval.game_kits.board_game.tictactoe.environment import TicTacToeArenaEnvironment


def test_tictactoe_arena_exposes_get_last_frame() -> None:
    env = TicTacToeArenaEnvironment(
        board_size=3,
        player_ids=["player_0", "player_1"],
        player_names={"player_0": "X", "player_1": "O"},
        coord_scheme="ROW_COL",
    )

    initial_frame = env.get_last_frame()
    assert isinstance(initial_frame, dict)
    assert initial_frame["active_player_id"] == "player_0"
    assert initial_frame["move_count"] == 0
    assert "board_text" in initial_frame
    assert isinstance(initial_frame["legal_moves"], list)

    env.observe("player_0")
    latest_frame = env.get_last_frame()
    assert latest_frame["observer_player_id"] == "player_0"
    assert latest_frame["active_player_id"] == "player_0"


def test_tictactoe_terminal_result_and_frame_expose_winning_line() -> None:
    env = TicTacToeArenaEnvironment(
        board_size=3,
        player_ids=["X", "O"],
        player_names={"X": "X", "O": "O"},
        coord_scheme="ROW_COL",
    )

    moves = [
        ("X", "1,1"),
        ("O", "1,2"),
        ("X", "2,2"),
        ("O", "1,3"),
        ("X", "3,3"),
    ]

    result = None
    for player_id, coord in moves:
        result = env.apply(
            ArenaAction(
                player=player_id,
                move=coord,
                raw=coord,
            )
        )

    assert result is not None
    assert getattr(result, "winning_line", None) == ["1,1", "2,2", "3,3"]

    final_frame = env.get_last_frame()
    assert final_frame["winning_line"] == ["1,1", "2,2", "3,3"]
    assert final_frame["last_move"] == "3,3"
