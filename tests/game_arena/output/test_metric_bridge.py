from __future__ import annotations

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.evaluation.cache import EvalCache
from gage_eval.evaluation.sample_envelope import append_arena_contract, ensure_arena_header
from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.arena import (
    EpisodeLengthStepsMetric,
    FinalScorePerPlayerMetric,
    IllegalReasonDistributionMetric,
)
from gage_eval.reporting.summary_generators.arena import ArenaSummaryGenerator


def test_arena_contract_fields_bridge_to_metrics_and_reporting(tmp_path) -> None:
    sample = {
        "id": "bridge-sample",
        "metadata": {"player_ids": ["p0", "p1"]},
        "predict_result": [{"index": 0, "answer": "stale"}],
    }
    ensure_arena_header(sample, start_time_ms=1000)

    model_output = {
        "sample": {"game_kit": "gomoku", "env": "gomoku_standard"},
        "result": {
            "winner": "p0",
            "reason": "finished",
            "move_count": 2,
            "final_scores": {"p0": 1, "p1": 0},
        },
        "arena_trace": [
            {
                "step_index": 99,
                "trace_state": "done",
                "timestamp": 1,
                "player_id": "stale",
                "action_raw": "Z9",
                "action_applied": "Z9",
                "t_obs_ready_ms": 1,
                "t_action_submitted_ms": 1,
                "timeout": False,
                "is_action_legal": True,
                "retry_count": 0,
            }
        ],
        "trace": [
            {
                "step_index": 0,
                "trace_state": "done",
                "timestamp": 1001,
                "player_id": "p0",
                "action_raw": "A1",
                "action_applied": "A1",
                "t_obs_ready_ms": 1001,
                "t_action_submitted_ms": 1002,
                "timeout": False,
                "is_action_legal": False,
                "retry_count": 1,
                "illegal_reason": "blocked",
            },
            {
                "step_index": 1,
                "trace_state": "done",
                "timestamp": 1003,
                "player_id": "p1",
                "action_raw": "B1",
                "action_applied": "B1",
                "t_obs_ready_ms": 1003,
                "t_action_submitted_ms": 1005,
                "timeout": False,
                "is_action_legal": True,
                "retry_count": 0,
            },
        ],
        "footer": {
            "winner_player_id": "p0",
            "termination_reason": "finished",
            "total_steps": 2,
            "final_scores": {"p0": 1, "p1": 0},
            "episode_returns": {"p0": 1.0, "p1": 0.0},
        },
        "artifacts": {"replay_ref": "runs/bridge-sample/replay.json"},
    }

    append_arena_contract(sample, model_output, end_time_ms=2000)

    entry = sample["predict_result"][0]
    assert [step["step_index"] for step in entry["arena_trace"]] == [0, 1]
    assert entry["trace"] == entry["arena_trace"]
    assert entry["artifacts"]["replay_ref"] == "runs/bridge-sample/replay.json"

    context = MetricContext(
        sample_id="bridge-sample",
        sample=sample,
        model_output={},
        judge_output={},
        args={},
        trace=None,
    )
    assert EpisodeLengthStepsMetric(
        MetricSpec("episode_length_steps", "episode_length_steps", None, {})
    ).compute(context).values == {"episode_length_steps": 2.0}
    assert FinalScorePerPlayerMetric(
        MetricSpec("final_score_per_player", "final_score_per_player", None, {})
    ).compute(context).values == {"p0": 1.0, "p1": 0.0}
    assert IllegalReasonDistributionMetric(
        MetricSpec("illegal_reason_distribution", "illegal_reason_distribution", None, {})
    ).compute(context).values == {"blocked": 1.0}

    cache = EvalCache(base_dir=str(tmp_path), run_id="arena-contract-bridge")
    cache.write_sample("bridge-sample", {"sample": sample})
    summary = ArenaSummaryGenerator().generate(cache)

    assert summary is not None
    payload = summary["arena_summary"]
    assert payload["overall"]["samples"] == 1
    assert payload["winner_player_id"] == {"p0": 1}
    assert payload["termination_reason"] == {"finished": 1}
