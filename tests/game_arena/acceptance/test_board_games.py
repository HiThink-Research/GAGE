from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("config_path", "game_kit", "env", "winner", "move_count"),
    [
        (
            REPO_ROOT / "config/custom/gomoku/gomoku_dummy_gamekit.yaml",
            "gomoku",
            "gomoku_standard",
            "Black",
            5,
        ),
        (
            REPO_ROOT / "config/custom/tictactoe/tictactoe_dummy_gamekit.yaml",
            "tictactoe",
            "tictactoe_standard",
            "X",
            5,
        ),
    ],
)
def test_board_game_gamekit_runs_dummy_match_end_to_end(
    run_gamearena_config,
    config_path: Path,
    game_kit: str,
    env: str,
    winner: str,
    move_count: int,
) -> None:
    result = run_gamearena_config(config_path)
    sample = result["sample"]
    output = result["output"]
    replay_path = Path(output["result"]["replay_path"])

    assert output["sample"]["game_kit"] == game_kit
    assert output["sample"]["env"] == env
    assert output["tick"] == move_count
    assert output["step"] == move_count
    assert output["result"]["winner"] == winner
    assert output["result"]["result"] == "win"
    assert output["result"]["move_count"] == move_count
    assert replay_path.exists()
    assert replay_path.name == "replay.json"
    assert replay_path.parent.parent.name == "replays"
    assert replay_path.parent.name == str(sample["id"])
    assert len(output["arena_trace"]) == move_count
    assert sample["predict_result"][0]["artifacts"]["replay_ref"] == str(replay_path)
    assert sample["predict_result"][0]["game_arena"]["winner_player_id"] == winner
    assert sample["predict_result"][0]["game_arena"]["total_steps"] == move_count
    assert sample["predict_result"][0]["arena_trace"] == list(output["arena_trace"])
