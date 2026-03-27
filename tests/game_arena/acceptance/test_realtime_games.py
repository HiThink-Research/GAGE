from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_vizdoom_gamekit_runs_dummy_match_end_to_end(
    run_gamearena_config,
) -> None:
    result = run_gamearena_config(
        REPO_ROOT / "config/custom/vizdoom/vizdoom_dummy_gamekit.yaml"
    )
    sample = result["sample"]
    output = result["output"]
    replay_path = Path(output["result"]["replay_path"])

    assert output["sample"]["game_kit"] == "vizdoom"
    assert output["sample"]["env"] == "duel_map01"
    assert output["sample"]["scheduler"] is None
    assert "scheduler" not in output["sample"]["runtime_overrides"]
    assert output["tick"] == output["step"] == output["result"]["move_count"] == 3
    assert output["result"]["result"] in {"win", "draw", "terminated"}
    assert replay_path.exists()
    assert output["arena_trace"]
    assert sample["predict_result"][0]["arena_trace"] == list(output["arena_trace"])


def test_retro_mario_gamekit_runs_dummy_match_end_to_end(
    run_gamearena_config,
) -> None:
    result = run_gamearena_config(
        REPO_ROOT / "config/custom/retro_mario/retro_mario_dummy_gamekit.yaml"
    )
    sample = result["sample"]
    output = result["output"]
    replay_path = Path(output["result"]["replay_path"])

    assert output["sample"]["game_kit"] == "retro_platformer"
    assert output["sample"]["env"] == "retro_mario"
    assert output["sample"]["scheduler"] is None
    assert "scheduler" not in output["sample"]["runtime_overrides"]
    assert output["tick"] > 0
    assert output["step"] > 0
    assert output["result"]["result"] in {"win", "draw", "terminated"}
    assert output["result"]["move_count"] > 0
    assert replay_path.exists()
    assert output["arena_trace"]
    assert sample["predict_result"][0]["arena_trace"] == list(output["arena_trace"])
