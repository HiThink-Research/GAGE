from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("config_path", "game_kit", "env", "winner", "move_count", "scheduler_override"),
    [
        (
            REPO_ROOT / "config/custom/doudizhu/doudizhu_dummy_gamekit.yaml",
            "doudizhu",
            "classic_3p",
            "landlord",
            4,
            None,
        ),
        (
            REPO_ROOT / "config/custom/mahjong/mahjong_dummy_gamekit.yaml",
            "mahjong",
            "riichi_4p",
            "east",
            5,
            None,
        ),
    ],
)
def test_phase_card_gamekits_run_dummy_match_end_to_end(
    run_gamearena_config,
    config_path: Path,
    game_kit: str,
    env: str,
    winner: str,
    move_count: int,
    scheduler_override: str | None,
) -> None:
    result = run_gamearena_config(config_path)
    sample = result["sample"]
    output = result["output"]
    replay_path = Path(output["result"]["replay_path"])

    assert output["sample"]["game_kit"] == game_kit
    assert output["sample"]["env"] == env
    assert output["sample"]["scheduler"] is None
    if scheduler_override is None:
        assert "scheduler" not in output["sample"]["runtime_overrides"]
    else:
        assert output["sample"]["runtime_overrides"]["scheduler"] == scheduler_override
    assert output["tick"] == output["step"] == move_count
    assert output["result"]["winner"] == winner
    assert output["result"]["result"] == "win"
    assert output["result"]["move_count"] == move_count
    assert output["result"]["reason"] in {"completed", "terminal"}
    assert output["arena_trace"]
    assert len(output["arena_trace"]) == move_count
    assert replay_path.exists()
    assert replay_path.name == "replay.json"
    assert replay_path.parent.parent.name == "replays"
    assert replay_path.parent.name == str(sample["id"])
    assert sample["predict_result"][0]["game_arena"]["winner_player_id"] == winner
    assert sample["predict_result"][0]["game_arena"]["total_steps"] == move_count
    assert sample["predict_result"][0]["arena_trace"] == list(output["arena_trace"])
