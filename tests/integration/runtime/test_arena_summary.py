from __future__ import annotations

from pathlib import Path

from gage_eval.evaluation.cache import EvalCache
from gage_eval.reporting.summary_generators.arena import (
    ArenaSummaryGenerator,
    _arena_entry,
    _build_arena_summary,
    _coerce_float,
    _duration_ms,
    _illegal_reason_distribution,
)


def _make_sample(sample_id: str, winner: str | None, reason: str, illegal_reason: str | None) -> dict:
    trace = [
        {
            "step_index": 0,
            "trace_state": "done",
            "timestamp": 1000,
            "player_id": "p0",
            "action_raw": "a",
            "action_applied": "a",
            "t_obs_ready_ms": 1000,
            "t_action_submitted_ms": 1002,
            "timeout": False,
            "is_action_legal": illegal_reason is None,
            "retry_count": 0,
        }
    ]
    if illegal_reason:
        trace[0]["illegal_reason"] = illegal_reason

    return {
        "id": sample_id,
        "metadata": {
            "game_arena": {
                "engine_id": "retro_env_v1",
                "seed": 1,
                "mode": "competitive",
                "players": [
                    {"player_id": "p0", "controller_type": "agent", "model_id": None, "policy_id": None},
                    {"player_id": "p1", "controller_type": "agent", "model_id": None, "policy_id": None},
                ],
                "start_time_ms": 1000,
            }
        },
        "predict_result": [
            {
                "index": 0,
                "arena_trace": trace,
                "game_arena": {
                    "end_time_ms": 2000,
                    "total_steps": 1,
                    "winner_player_id": winner,
                    "termination_reason": reason,
                    "ranks": ["p0", "p1"],
                },
            }
        ],
    }


def test_arena_summary_generator(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=str(tmp_path), run_id="arena-summary")
    cache.write_sample(
        "s1",
        {"sample": _make_sample("s1", "p0", "finished", "blocked")},
    )
    cache.write_sample(
        "s2",
        {"sample": _make_sample("s2", None, "timeout", None)},
    )

    summary = ArenaSummaryGenerator().generate(cache)

    assert summary is not None
    payload = summary["arena_summary"]
    assert payload["overall"]["samples"] == 2
    assert payload["overall"]["avg_episode_duration_ms"] == 1000.0
    assert payload["winner_player_id"]["p0"] == 1
    assert payload["termination_reason"]["finished"] == 1
    assert payload["termination_reason"]["timeout"] == 1
    assert payload["illegal_reason_distribution"]["blocked"] == 1


def test_arena_summary_handles_empty_and_invalid_records(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=str(tmp_path), run_id="arena-summary-edge")
    cache.write_sample("raw", {"not_sample": True})
    cache.write_sample("bad1", {"sample": {"id": "x"}})
    cache.write_sample("bad2", {"sample": {"predict_result": [{"index": 0, "arena_trace": []}]}})

    assert ArenaSummaryGenerator().generate(cache) is None
    assert _build_arena_summary(cache) is None


def test_arena_summary_helper_paths(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=str(tmp_path), run_id="arena-summary-helper")
    sample = _make_sample("s3", "p1", "finished", None)
    sample["predict_result"][0]["game_arena"]["ranks"] = [{"player_id": "p1", "rank": 1}]
    sample["predict_result"].append(
        {
            "index": 1,
            "arena_trace": sample["predict_result"][0]["arena_trace"],
            "game_arena": {
                "end_time_ms": 2000,
                "total_steps": 1,
                "winner_player_id": "p0",
                "termination_reason": "finished",
                "ranks": ["p0", "p1"],
            },
        }
    )
    sample["selected_predict_result_index"] = 1
    cache.write_sample("s3", {"sample": sample})

    payload = _build_arena_summary(cache)
    assert payload is not None
    assert payload["rank_top1"] == {"p0": 1}

    assert _arena_entry({"predict_result": []}) == {}
    assert _arena_entry({"predict_result": [None]}) == {}
    assert _arena_entry(sample)["index"] == 1
    assert _coerce_float("3.5") == 3.5
    assert _coerce_float("x") is None

    entry = {"arena_trace": [{"illegal_reason": "blocked"}, {"illegal_reason": ""}, None]}
    assert _illegal_reason_distribution(entry) == {"blocked": 1}
    assert _illegal_reason_distribution({"arena_trace": "bad"}) == {}

    assert _duration_ms({}, {}) is None
    assert _duration_ms({"metadata": {"game_arena": {}}}, {"end_time_ms": 1}) is None
    assert (
        _duration_ms({"metadata": {"game_arena": {"start_time_ms": 5}}}, {"end_time_ms": 3})
        == 0.0
    )
