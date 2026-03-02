from __future__ import annotations

import pytest

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.arena import (
    CompletionFlagMetric,
    DrawFlagMetric,
    EpisodeDurationMsMetric,
    EpisodeLengthStepsMetric,
    FinalScorePerPlayerMetric,
    IllegalActionCountMetric,
    IllegalReasonDistributionMetric,
    LegalActionRateMetric,
    ObsToActionLatencyMeanMetric,
    ObsToActionLatencyP50Metric,
    ObsToActionLatencyP95Metric,
    OnTimeRateMetric,
    RankListMetric,
    RewardPerSecondPerPlayerMetric,
    RetryCountMeanMetric,
    RetryCountP95Metric,
    ScoreMarginMetric,
    TerminationReasonMetric,
    TimeoutCountMetric,
    WinnerPlayerIdMetric,
    WinFlagPerPlayerMetric,
    _coerce_bool,
    _coerce_float_map,
    _percentile,
)


def _build_sample() -> dict:
    return {
        "id": "arena-sample",
        "metadata": {
            "game_arena": {
                "engine_id": "retro_env_v1",
                "seed": 42,
                "mode": "competitive",
                "players": [
                    {"player_id": "p0", "controller_type": "agent", "model_id": "m0", "policy_id": None},
                    {"player_id": "p1", "controller_type": "agent", "model_id": "m1", "policy_id": None},
                ],
                "start_time_ms": 1000,
            }
        },
        "predict_result": [
            {
                "index": 0,
                "arena_trace": [
                    {
                        "step_index": 0,
                        "trace_state": "done",
                        "timestamp": 1001,
                        "player_id": "p0",
                        "action_raw": "a",
                        "action_applied": "a",
                        "t_obs_ready_ms": 1001,
                        "t_action_submitted_ms": 1003,
                        "timeout": False,
                        "is_action_legal": True,
                        "retry_count": 0,
                    },
                    {
                        "step_index": 1,
                        "trace_state": "done",
                        "timestamp": 1004,
                        "player_id": "p1",
                        "action_raw": "b",
                        "action_applied": "b",
                        "t_obs_ready_ms": 1004,
                        "t_action_submitted_ms": 1008,
                        "timeout": True,
                        "deadline_ms": 10,
                        "is_action_legal": False,
                        "retry_count": 1,
                        "illegal_reason": "blocked",
                    },
                    {
                        "step_index": 2,
                        "trace_state": "done",
                        "timestamp": 1009,
                        "player_id": "p0",
                        "action_raw": "c",
                        "action_applied": "c",
                        "t_obs_ready_ms": 1009,
                        "t_action_submitted_ms": 1015,
                        "timeout": False,
                        "deadline_ms": 10,
                        "is_action_legal": True,
                        "retry_count": 0,
                    },
                    {
                        "step_index": 3,
                        "trace_state": "done",
                        "timestamp": 1016,
                        "player_id": "p1",
                        "action_raw": "d",
                        "action_applied": "d",
                        "t_obs_ready_ms": 1016,
                        "t_action_submitted_ms": 1024,
                        "timeout": False,
                        "is_action_legal": True,
                        "retry_count": 2,
                    },
                ],
                "game_arena": {
                    "end_time_ms": 5000,
                    "total_steps": 4,
                    "winner_player_id": "p0",
                    "termination_reason": "finished",
                    "ranks": ["p0", "p1"],
                    "final_scores": {"p0": 10, "p1": 6},
                    "episode_returns": {"p0": 2.5, "p1": 1.0},
                },
            }
        ],
    }


def _ctx(sample: dict, mock_trace) -> MetricContext:
    return MetricContext(
        sample_id="arena-sample",
        sample=sample,
        model_output={},
        judge_output={},
        args={},
        trace=mock_trace,
    )


def test_arena_timing_metrics(mock_trace) -> None:
    context = _ctx(_build_sample(), mock_trace)

    duration_metric = EpisodeDurationMsMetric(MetricSpec("episode_duration_ms", "episode_duration_ms", None, {}))
    length_metric = EpisodeLengthStepsMetric(MetricSpec("episode_length_steps", "episode_length_steps", None, {}))
    mean_metric = ObsToActionLatencyMeanMetric(
        MetricSpec("obs_to_action_latency_ms_mean", "obs_to_action_latency_ms_mean", None, {})
    )
    p50_metric = ObsToActionLatencyP50Metric(
        MetricSpec("obs_to_action_latency_ms_p50", "obs_to_action_latency_ms_p50", None, {})
    )
    p95_metric = ObsToActionLatencyP95Metric(
        MetricSpec("obs_to_action_latency_ms_p95", "obs_to_action_latency_ms_p95", None, {})
    )

    assert duration_metric.compute(context).values["episode_duration_ms"] == 4000.0
    assert length_metric.compute(context).values["episode_length_steps"] == 4.0
    assert mean_metric.compute(context).values["obs_to_action_latency_ms_mean"] == pytest.approx(5.0)
    assert p50_metric.compute(context).values["obs_to_action_latency_ms_p50"] == pytest.approx(5.0)
    assert p95_metric.compute(context).values["obs_to_action_latency_ms_p95"] == pytest.approx(7.7)


def test_arena_legality_and_timeout_metrics(mock_trace) -> None:
    context = _ctx(_build_sample(), mock_trace)

    on_time_metric = OnTimeRateMetric(MetricSpec("on_time_rate", "on_time_rate", None, {}))
    timeout_metric = TimeoutCountMetric(MetricSpec("timeout_count", "timeout_count", None, {}))
    legal_rate_metric = LegalActionRateMetric(MetricSpec("legal_action_rate", "legal_action_rate", None, {}))
    illegal_count_metric = IllegalActionCountMetric(
        MetricSpec("illegal_action_count", "illegal_action_count", None, {})
    )
    retry_mean_metric = RetryCountMeanMetric(MetricSpec("retry_count_mean", "retry_count_mean", None, {}))
    retry_p95_metric = RetryCountP95Metric(MetricSpec("retry_count_p95", "retry_count_p95", None, {}))

    assert on_time_metric.compute(context).values["on_time_rate"] == pytest.approx(0.5)
    assert timeout_metric.compute(context).values["timeout_count"] == 1.0
    assert legal_rate_metric.compute(context).values["legal_action_rate"] == pytest.approx(0.75)
    assert illegal_count_metric.compute(context).values["illegal_action_count"] == 1.0
    assert retry_mean_metric.compute(context).values["retry_count_mean"] == pytest.approx(0.75)
    assert retry_p95_metric.compute(context).values["retry_count_p95"] == pytest.approx(1.85)


def test_arena_competitive_metrics(mock_trace) -> None:
    context = _ctx(_build_sample(), mock_trace)

    draw_metric = DrawFlagMetric(MetricSpec("draw_flag", "draw_flag", None, {}))
    completion_metric = CompletionFlagMetric(MetricSpec("completion_flag", "completion_flag", None, {}))
    margin_metric = ScoreMarginMetric(MetricSpec("score_margin", "score_margin", None, {}))
    win_flag_metric = WinFlagPerPlayerMetric(MetricSpec("win_flag_per_player", "win_flag_per_player", None, {}))

    assert draw_metric.compute(context).values["draw_flag"] == 0.0
    assert completion_metric.compute(context).values["completion_flag"] == 1.0
    assert margin_metric.compute(context).values["score_margin"] == 4.0
    assert win_flag_metric.compute(context).values == {"p0": 1.0, "p1": 0.0}


def test_arena_footer_and_distribution_metrics(mock_trace) -> None:
    context = _ctx(_build_sample(), mock_trace)

    score_metric = FinalScorePerPlayerMetric(MetricSpec("final_score_per_player", "final_score_per_player", None, {}))
    rps_metric = RewardPerSecondPerPlayerMetric(
        MetricSpec("reward_per_second_per_player", "reward_per_second_per_player", None, {})
    )
    illegal_reason_metric = IllegalReasonDistributionMetric(
        MetricSpec("illegal_reason_distribution", "illegal_reason_distribution", None, {})
    )
    winner_metric = WinnerPlayerIdMetric(MetricSpec("winner_player_id", "winner_player_id", None, {}))
    rank_metric = RankListMetric(MetricSpec("rank_list", "rank_list", None, {}))
    term_metric = TerminationReasonMetric(MetricSpec("termination_reason", "termination_reason", None, {}))

    assert score_metric.compute(context).values == {"p0": 10.0, "p1": 6.0}
    assert rps_metric.compute(context).values == {"p0": pytest.approx(0.625), "p1": pytest.approx(0.25)}
    assert illegal_reason_metric.compute(context).values == {"blocked": 1.0}
    winner_result = winner_metric.compute(context)
    assert winner_result.values == {"winner_present": 1.0}
    assert winner_result.metadata == {"winner_player_id": "p0"}
    assert rank_metric.compute(context).values == {"p0": 1.0, "p1": 2.0}
    term_result = term_metric.compute(context)
    assert term_result.values == {"termination_present": 1.0}
    assert term_result.metadata == {"termination_reason": "finished"}


def test_arena_metrics_fallback_paths(mock_trace) -> None:
    sample = {
        "id": "edge",
        "metadata": {"game_arena": {"start_time_ms": "bad", "players": "bad-shape"}},
        "predict_result": [
            {
                "index": 0,
                "arena_trace": "not-a-list",
                "game_arena": {
                    "end_time_ms": "oops",
                    "total_steps": "oops",
                    "winner_player_id": "",
                    "termination_reason": "",
                    "ranks": [{"player_id": "p0", "rank": "1"}, {"player_id": "p1", "rank": "x"}, None, "p2"],
                    "episode_returns": {"p0": "3", "p1": "bad"},
                },
            }
        ],
    }
    context = _ctx(sample, mock_trace)

    duration_metric = EpisodeDurationMsMetric(MetricSpec("episode_duration_ms", "episode_duration_ms", None, {}))
    length_metric = EpisodeLengthStepsMetric(MetricSpec("episode_length_steps", "episode_length_steps", None, {}))
    on_time_metric = OnTimeRateMetric(MetricSpec("on_time_rate", "on_time_rate", None, {}))
    draw_metric = DrawFlagMetric(MetricSpec("draw_flag", "draw_flag", None, {}))
    margin_metric = ScoreMarginMetric(MetricSpec("score_margin", "score_margin", None, {}))
    win_metric = WinFlagPerPlayerMetric(MetricSpec("win_flag_per_player", "win_flag_per_player", None, {}))
    rank_metric = RankListMetric(MetricSpec("rank_list", "rank_list", None, {}))
    completion_metric = CompletionFlagMetric(MetricSpec("completion_flag", "completion_flag", None, {}))
    winner_metric = WinnerPlayerIdMetric(MetricSpec("winner_player_id", "winner_player_id", None, {}))
    term_metric = TerminationReasonMetric(MetricSpec("termination_reason", "termination_reason", None, {}))

    assert duration_metric.compute(context).values["episode_duration_ms"] == 0.0
    assert length_metric.compute(context).values["episode_length_steps"] == 0.0
    assert on_time_metric.compute(context).values["on_time_rate"] == 0.0
    assert draw_metric.compute(context).values["draw_flag"] == 1.0
    assert margin_metric.compute(context).values["score_margin"] == 0.0
    assert win_metric.compute(context).values == {}
    assert rank_metric.compute(context).values == {"p0": 1.0, "p2": 4.0}
    assert completion_metric.compute(context).values["completion_flag"] == 0.0
    assert winner_metric.compute(context).metadata["winner_player_id"] == ""
    assert term_metric.compute(context).metadata["termination_reason"] == ""

    assert _coerce_float_map(None) is None
    assert _coerce_float_map({"a": "1", "b": "x"}) == {"a": 1.0}
    assert _coerce_bool("on", default=False) is True
    assert _coerce_bool("off", default=True) is False
    assert _coerce_bool(0, default=True) is False
    assert _percentile([], 95.0) == 0.0
    assert _percentile([3.0], 95.0) == 3.0
