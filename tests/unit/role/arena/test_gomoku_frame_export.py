from __future__ import annotations

import base64

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


def test_gomoku_arena_observe_includes_encoded_image_when_obs_image_enabled() -> None:
    env = GomokuArenaEnvironment(
        board_size=5,
        win_len=4,
        player_ids=["black", "white"],
        player_names={"black": "Black", "white": "White"},
        coord_scheme="A1",
        obs_image=True,
    )

    observation = env.observe("black")

    image = (observation.view or {}).get("image")
    assert isinstance(image, dict)
    assert str(image.get("data_url")).startswith("data:image/png;base64,")
    assert int(image.get("width", 0)) > 0
    assert int(image.get("height", 0)) > 0
    _, _, payload = str(image.get("data_url")).partition(",")
    assert len(base64.b64decode(payload)) > 0
