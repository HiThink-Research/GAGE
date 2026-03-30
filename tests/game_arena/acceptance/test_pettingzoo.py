from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_pettingzoo_gamekit_runs_dummy_match_end_to_end(
    run_gamearena_config,
) -> None:
    result = run_gamearena_config(
        REPO_ROOT / "config/custom/pettingzoo/space_invaders_dummy_gamekit.yaml"
    )
    sample = result["sample"]
    output = result["output"]
    replay_path = Path(output["result"]["replay_path"])

    assert output["sample"]["game_kit"] == "pettingzoo"
    assert output["sample"]["env"] == "space_invaders"
    assert output["sample"]["scheduler"] is None
    assert "scheduler" not in output["sample"]["runtime_overrides"]
    assert output["tick"] == 4
    assert output["step"] == 4
    assert output["result"]["winner"] == "pilot_alpha"
    assert output["result"]["result"] == "win"
    assert output["result"]["move_count"] == 4
    assert replay_path.exists()
    assert replay_path.name == "replay.json"
    assert replay_path.parent.parent.name == "replays"
    assert replay_path.parent.name == str(sample["id"])
    assert len(output["arena_trace"]) == 4
    assert sample["predict_result"][0]["artifacts"]["replay_ref"] == str(replay_path)
    assert sample["predict_result"][0]["game_arena"]["winner_player_id"] == "pilot_alpha"
    assert sample["predict_result"][0]["game_arena"]["total_steps"] == 4
    assert sample["predict_result"][0]["arena_trace"] == list(output["arena_trace"])
