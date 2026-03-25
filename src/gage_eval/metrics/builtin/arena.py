"""Arena metrics based on the frozen trace/footer contract."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from gage_eval.evaluation.sample_envelope import (
    resolve_arena_trace,
    resolve_selected_predict_result,
)
from gage_eval.metrics.base import BaseMetric, MetricContext, MetricResult, SimpleMetric
from gage_eval.registry import registry


@registry.asset(
    "metrics",
    "final_score_per_player",
    desc="Reports final score values per player",
    tags=("arena", "game"),
)
class FinalScorePerPlayerMetric(BaseMetric):
    """Reports final score values per player."""

    def compute(self, context: MetricContext) -> MetricResult:
        footer = _arena_footer(context)
        scores = _coerce_float_map(footer.get("final_scores")) or {}
        return MetricResult(sample_id=context.sample_id, values=scores)


@registry.asset(
    "metrics",
    "episode_duration_ms",
    desc="Computes arena episode wall-clock duration in milliseconds",
    tags=("arena", "game"),
)
class EpisodeDurationMsMetric(SimpleMetric):
    """Computes arena episode wall-clock duration in milliseconds."""

    value_key = "episode_duration_ms"

    def compute_value(self, context: MetricContext) -> float:
        footer = _arena_footer(context)
        header = _arena_header(context)
        start_ms = _coerce_float(header.get("start_time_ms"))
        end_ms = _coerce_float(footer.get("end_time_ms"))
        if start_ms is None or end_ms is None:
            return 0.0
        return max(0.0, end_ms - start_ms)


@registry.asset(
    "metrics",
    "episode_length_steps",
    desc="Computes arena episode decision step count",
    tags=("arena", "game"),
)
class EpisodeLengthStepsMetric(SimpleMetric):
    """Computes arena episode decision step count."""

    value_key = "episode_length_steps"

    def compute_value(self, context: MetricContext) -> float:
        footer = _arena_footer(context)
        total_steps = _coerce_float(footer.get("total_steps"))
        if total_steps is not None:
            return max(0.0, total_steps)
        return float(len(_arena_trace(context)))


@registry.asset(
    "metrics",
    "reward_per_second_per_player",
    desc="Computes reward-per-second values for each player",
    tags=("arena", "game"),
)
class RewardPerSecondPerPlayerMetric(BaseMetric):
    """Computes reward-per-second values for each player."""

    def compute(self, context: MetricContext) -> MetricResult:
        footer = _arena_footer(context)
        header = _arena_header(context)
        returns = _coerce_float_map(footer.get("episode_returns")) or {}
        start_ms = _coerce_float(header.get("start_time_ms"))
        end_ms = _coerce_float(footer.get("end_time_ms"))
        duration_s = 0.0
        if start_ms is not None and end_ms is not None and end_ms > start_ms:
            duration_s = (end_ms - start_ms) / 1000.0
        values: dict[str, float] = {}
        for player_id, ret in returns.items():
            values[player_id] = ret / duration_s if duration_s > 0.0 else 0.0
        return MetricResult(sample_id=context.sample_id, values=values)


@registry.asset(
    "metrics",
    "obs_to_action_latency_ms_mean",
    desc="Computes mean observation-to-action latency in milliseconds",
    tags=("arena", "latency"),
)
class ObsToActionLatencyMeanMetric(SimpleMetric):
    """Computes mean observation-to-action latency in milliseconds."""

    value_key = "obs_to_action_latency_ms_mean"

    def compute_value(self, context: MetricContext) -> float:
        latencies = _latencies_ms(_arena_trace(context))
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)


@registry.asset(
    "metrics",
    "obs_to_action_latency_ms_p50",
    desc="Computes p50 observation-to-action latency in milliseconds",
    tags=("arena", "latency"),
)
class ObsToActionLatencyP50Metric(SimpleMetric):
    """Computes p50 observation-to-action latency in milliseconds."""

    value_key = "obs_to_action_latency_ms_p50"

    def compute_value(self, context: MetricContext) -> float:
        return _percentile(_latencies_ms(_arena_trace(context)), 50.0)


@registry.asset(
    "metrics",
    "obs_to_action_latency_ms_p95",
    desc="Computes p95 observation-to-action latency in milliseconds",
    tags=("arena", "latency"),
)
class ObsToActionLatencyP95Metric(SimpleMetric):
    """Computes p95 observation-to-action latency in milliseconds."""

    value_key = "obs_to_action_latency_ms_p95"

    def compute_value(self, context: MetricContext) -> float:
        return _percentile(_latencies_ms(_arena_trace(context)), 95.0)


@registry.asset(
    "metrics",
    "on_time_rate",
    desc="Computes on-time action submission rate",
    tags=("arena", "latency"),
)
class OnTimeRateMetric(SimpleMetric):
    """Computes on-time action submission rate."""

    value_key = "on_time_rate"

    def compute_value(self, context: MetricContext) -> float:
        constrained = _constrained_timeout_steps(_arena_trace(context))
        if not constrained:
            return 0.0
        on_time = sum(1 for step in constrained if not _coerce_bool(step.get("timeout"), default=False))
        return on_time / float(len(constrained))


@registry.asset(
    "metrics",
    "timeout_count",
    desc="Counts timed-out action submissions",
    tags=("arena", "latency"),
)
class TimeoutCountMetric(SimpleMetric):
    """Counts timed-out action submissions."""

    value_key = "timeout_count"

    def compute_value(self, context: MetricContext) -> float:
        constrained = _constrained_timeout_steps(_arena_trace(context))
        timeout_count = sum(1 for step in constrained if _coerce_bool(step.get("timeout"), default=False))
        return float(timeout_count)


@registry.asset(
    "metrics",
    "legal_action_rate",
    desc="Computes legal action ratio",
    tags=("arena", "legality"),
)
class LegalActionRateMetric(SimpleMetric):
    """Computes legal action ratio."""

    value_key = "legal_action_rate"

    def compute_value(self, context: MetricContext) -> float:
        trace_steps = _arena_trace(context)
        if not trace_steps:
            return 0.0
        legal_count = sum(1 for step in trace_steps if _coerce_bool(step.get("is_action_legal"), default=True))
        return legal_count / float(len(trace_steps))


@registry.asset(
    "metrics",
    "illegal_action_count",
    desc="Counts illegal actions",
    tags=("arena", "legality"),
)
class IllegalActionCountMetric(SimpleMetric):
    """Counts illegal actions."""

    value_key = "illegal_action_count"

    def compute_value(self, context: MetricContext) -> float:
        trace_steps = _arena_trace(context)
        illegal_count = sum(1 for step in trace_steps if not _coerce_bool(step.get("is_action_legal"), default=True))
        return float(illegal_count)


@registry.asset(
    "metrics",
    "retry_count_mean",
    desc="Computes mean retry count per step",
    tags=("arena", "legality"),
)
class RetryCountMeanMetric(SimpleMetric):
    """Computes mean retry count per step."""

    value_key = "retry_count_mean"

    def compute_value(self, context: MetricContext) -> float:
        retries = _retry_counts(_arena_trace(context))
        if not retries:
            return 0.0
        return sum(retries) / float(len(retries))


@registry.asset(
    "metrics",
    "retry_count_p95",
    desc="Computes p95 retry count per step",
    tags=("arena", "legality"),
)
class RetryCountP95Metric(SimpleMetric):
    """Computes p95 retry count per step."""

    value_key = "retry_count_p95"

    def compute_value(self, context: MetricContext) -> float:
        return _percentile(_retry_counts(_arena_trace(context)), 95.0)


@registry.asset(
    "metrics",
    "illegal_reason_distribution",
    desc="Reports illegal reason counts per sample",
    tags=("arena", "legality"),
)
class IllegalReasonDistributionMetric(BaseMetric):
    """Reports illegal reason counts per sample."""

    def compute(self, context: MetricContext) -> MetricResult:
        counts: dict[str, float] = {}
        for step in _arena_trace(context):
            reason = step.get("illegal_reason")
            if reason in (None, ""):
                continue
            key = str(reason)
            counts[key] = counts.get(key, 0.0) + 1.0
        return MetricResult(sample_id=context.sample_id, values=counts)


@registry.asset(
    "metrics",
    "winner_player_id",
    desc="Emits winner metadata for categorical aggregation",
    tags=("arena", "competitive"),
)
class WinnerPlayerIdMetric(SimpleMetric):
    """Emits winner metadata for categorical aggregation."""

    value_key = "winner_present"

    def compute_value(self, context: MetricContext) -> tuple[float, dict[str, Any]]:
        footer = _arena_footer(context)
        winner = footer.get("winner_player_id")
        return (1.0 if winner not in (None, "") else 0.0), {"winner_player_id": winner}


@registry.asset(
    "metrics",
    "win_flag_per_player",
    desc="Reports per-player win flags",
    tags=("arena", "competitive"),
)
class WinFlagPerPlayerMetric(BaseMetric):
    """Reports per-player win flags."""

    def compute(self, context: MetricContext) -> MetricResult:
        header = _arena_header(context)
        footer = _arena_footer(context)
        winner = footer.get("winner_player_id")
        values: dict[str, float] = {}
        players = header.get("players")
        if isinstance(players, Sequence) and not isinstance(players, (str, bytes)):
            for item in players:
                if not isinstance(item, Mapping):
                    continue
                player_id = item.get("player_id")
                if not player_id:
                    continue
                key = str(player_id)
                values[key] = 1.0 if winner == key else 0.0
        return MetricResult(sample_id=context.sample_id, values=values)


@registry.asset(
    "metrics",
    "draw_flag",
    desc="Computes draw flag",
    tags=("arena", "competitive"),
)
class DrawFlagMetric(SimpleMetric):
    """Computes draw flag."""

    value_key = "draw_flag"

    def compute_value(self, context: MetricContext) -> float:
        footer = _arena_footer(context)
        winner = footer.get("winner_player_id")
        return 1.0 if winner in (None, "") else 0.0


@registry.asset(
    "metrics",
    "rank_list",
    desc="Reports rank positions per player",
    tags=("arena", "competitive"),
)
class RankListMetric(BaseMetric):
    """Reports rank positions per player."""

    def compute(self, context: MetricContext) -> MetricResult:
        footer = _arena_footer(context)
        values: dict[str, float] = {}
        ranks = footer.get("ranks")
        if isinstance(ranks, Sequence) and not isinstance(ranks, (str, bytes)):
            for idx, item in enumerate(ranks):
                if isinstance(item, Mapping):
                    player_id = item.get("player_id")
                    rank_value = _coerce_float(item.get("rank"))
                    if player_id and rank_value is not None:
                        values[str(player_id)] = rank_value
                    continue
                if item in (None, ""):
                    continue
                values[str(item)] = float(idx + 1)
        return MetricResult(sample_id=context.sample_id, values=values)


@registry.asset(
    "metrics",
    "score_margin",
    desc="Computes score margin between first and second place",
    tags=("arena", "competitive"),
)
class ScoreMarginMetric(SimpleMetric):
    """Computes score margin between first and second place."""

    value_key = "score_margin"

    def compute_value(self, context: MetricContext) -> float:
        footer = _arena_footer(context)
        scores = _coerce_float_map(footer.get("final_scores"))
        if not scores:
            scores = _coerce_float_map(footer.get("episode_returns"))
        if not scores or len(scores) < 2:
            return 0.0
        sorted_scores = sorted(scores.values(), reverse=True)
        return sorted_scores[0] - sorted_scores[1]


@registry.asset(
    "metrics",
    "termination_reason",
    desc="Emits termination reason metadata for categorical aggregation",
    tags=("arena", "stability"),
)
class TerminationReasonMetric(SimpleMetric):
    """Emits termination reason metadata for categorical aggregation."""

    value_key = "termination_present"

    def compute_value(self, context: MetricContext) -> tuple[float, dict[str, Any]]:
        footer = _arena_footer(context)
        reason = footer.get("termination_reason")
        return (1.0 if reason not in (None, "") else 0.0), {"termination_reason": reason}


@registry.asset(
    "metrics",
    "completion_flag",
    desc="Computes completion flag (termination_reason == finished)",
    tags=("arena", "stability"),
)
class CompletionFlagMetric(SimpleMetric):
    """Computes completion flag (termination_reason == finished)."""

    value_key = "completion_flag"

    def compute_value(self, context: MetricContext) -> float:
        footer = _arena_footer(context)
        reason = str(footer.get("termination_reason") or "").lower()
        return 1.0 if reason == "finished" else 0.0


def _arena_entry(context: MetricContext) -> Mapping[str, Any]:
    sample = context.sample if isinstance(context.sample, Mapping) else {}
    return resolve_selected_predict_result(sample, domain="arena")


def _arena_trace(context: MetricContext) -> list[Mapping[str, Any]]:
    trace = resolve_arena_trace(context.sample, context.model_output)
    return [item for item in trace if isinstance(item, Mapping)]


def _arena_footer(context: MetricContext) -> Mapping[str, Any]:
    entry = _arena_entry(context)
    footer = entry.get("game_arena")
    if isinstance(footer, Mapping):
        return footer
    return {}


def _arena_header(context: MetricContext) -> Mapping[str, Any]:
    sample = context.sample if isinstance(context.sample, Mapping) else {}
    metadata = sample.get("metadata")
    if not isinstance(metadata, Mapping):
        return {}
    header = metadata.get("game_arena")
    if isinstance(header, Mapping):
        return header
    return {}


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_float_map(value: Any) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    parsed: dict[str, float] = {}
    for key, raw in value.items():
        number = _coerce_float(raw)
        if number is None:
            continue
        parsed[str(key)] = number
    return parsed or None


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _latencies_ms(trace_steps: Sequence[Mapping[str, Any]]) -> list[float]:
    latencies: list[float] = []
    for step in trace_steps:
        obs_ms = _coerce_float(step.get("t_obs_ready_ms"))
        action_ms = _coerce_float(step.get("t_action_submitted_ms"))
        if obs_ms is None or action_ms is None:
            continue
        latencies.append(max(0.0, action_ms - obs_ms))
    return latencies


def _retry_counts(trace_steps: Sequence[Mapping[str, Any]]) -> list[float]:
    retries: list[float] = []
    for step in trace_steps:
        value = _coerce_float(step.get("retry_count"))
        retries.append(max(0.0, value or 0.0))
    return retries


def _constrained_timeout_steps(trace_steps: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    constrained: list[Mapping[str, Any]] = []
    for step in trace_steps:
        if step.get("deadline_ms") is not None:
            constrained.append(step)
    if constrained:
        return constrained
    return list(trace_steps)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (max(0.0, min(100.0, percentile)) / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight
