from __future__ import annotations

import base64

from gage_eval.game_kits.board_game.gomoku.envs.gomoku_standard import (
    GomokuStandardEnvironment,
)
from gage_eval.game_kits.board_game.gomoku.environment import GomokuArenaEnvironment


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


def test_gomoku_gamekit_env_preserves_obs_image_runtime_override() -> None:
    class _Player:
        def __init__(self, player_id: str) -> None:
            self.player_id = player_id
            self.display_name = player_id

    env = GomokuStandardEnvironment(
        board_size=5,
        win_len=4,
        coord_scheme="A1",
        rule_profile="freestyle",
        win_directions=("horizontal", "vertical", "diagonal", "anti_diagonal"),
        illegal_policy=None,
        obs_image=True,
        player_specs=(_Player("black"), _Player("white")),
        start_player_id="black",
    )

    observation = env.observe("black")

    image = (observation.view or {}).get("image")
    assert isinstance(image, dict)
    assert str(image.get("data_url")).startswith("data:image/png;base64,")
