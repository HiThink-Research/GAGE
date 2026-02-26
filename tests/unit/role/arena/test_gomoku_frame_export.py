from __future__ import annotations

from gage_eval.role.arena.games.gomoku.env import GomokuArenaEnvironment


def test_gomoku_arena_exposes_get_last_frame() -> None:
    env = GomokuArenaEnvironment(
        board_size=5,
        win_len=4,
        player_ids=["black", "white"],
        player_names={"black": "Black", "white": "White"},
        coord_scheme="A1",
    )

    initial_frame = env.get_last_frame()
    assert isinstance(initial_frame, dict)
    assert initial_frame["active_player_id"] == "black"
    assert initial_frame["move_count"] == 0
    assert "board_text" in initial_frame
    assert isinstance(initial_frame["legal_moves"], list)

    env.observe("black")
    latest_frame = env.get_last_frame()
    assert latest_frame["observer_player_id"] == "black"
    assert latest_frame["active_player_id"] == "black"
