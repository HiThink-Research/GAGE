from __future__ import annotations

from gage_eval.role.arena.games.tictactoe.env import TicTacToeArenaEnvironment


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
