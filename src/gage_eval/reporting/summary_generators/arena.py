"""Arena summary generator."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from gage_eval.evaluation.cache import EvalCache
from gage_eval.evaluation.sample_envelope import resolve_selected_predict_result
from gage_eval.registry import registry
from gage_eval.reporting.summary_generators import SummaryGenerator


@registry.asset(
    "summary_generators",
    "arena_summary",
    desc="Arena summary generator",
    tags=("arena", "report"),
    default_enabled=True,
)
class ArenaSummaryGenerator(SummaryGenerator):
    """Builds arena aggregate summaries from cached samples."""

    def generate(self, cache: EvalCache) -> Optional[Dict[str, Any]]:
        summary = _build_arena_summary(cache)
        if not summary:
            return None
        return {"arena_summary": summary}


def _build_arena_summary(cache: EvalCache) -> Optional[Dict[str, Any]]:
    sample_count = 0
    duration_sum = 0.0
    duration_count = 0
    step_sum = 0.0
    step_count = 0
    draw_count = 0

    winner_counts: Dict[str, int] = {}
    termination_reason_counts: Dict[str, int] = {}
    illegal_reason_counts: Dict[str, int] = {}
    top1_counts: Dict[str, int] = {}

    for record in cache.iter_samples():
        sample = record.get("sample") if isinstance(record, Mapping) else None
        if not isinstance(sample, Mapping):
            continue
        entry = _arena_entry(sample)
        if not entry:
            continue
        footer = entry.get("game_arena")
        if not isinstance(footer, Mapping):
            continue
        sample_count += 1

        duration = _duration_ms(sample, footer)
        if duration is not None:
            duration_sum += duration
            duration_count += 1

        total_steps = _coerce_float(footer.get("total_steps"))
        if total_steps is not None:
            step_sum += total_steps
            step_count += 1

        winner = footer.get("winner_player_id")
        if winner in (None, ""):
            draw_count += 1
        else:
            winner_key = str(winner)
            winner_counts[winner_key] = winner_counts.get(winner_key, 0) + 1

        termination_reason = footer.get("termination_reason")
        if termination_reason not in (None, ""):
            reason_key = str(termination_reason)
            termination_reason_counts[reason_key] = termination_reason_counts.get(reason_key, 0) + 1

        for reason_key, reason_count in _illegal_reason_distribution(entry).items():
            illegal_reason_counts[reason_key] = illegal_reason_counts.get(reason_key, 0) + reason_count

        ranks = footer.get("ranks")
        if isinstance(ranks, Sequence) and not isinstance(ranks, (str, bytes)) and ranks:
            top_player = ranks[0]
            if isinstance(top_player, Mapping):
                top_player = top_player.get("player_id")
            if top_player not in (None, ""):
                player_key = str(top_player)
                top1_counts[player_key] = top1_counts.get(player_key, 0) + 1

    if sample_count == 0:
        return None

    return {
        "overall": {
            "samples": sample_count,
            "avg_episode_duration_ms": (duration_sum / duration_count) if duration_count else 0.0,
            "avg_episode_length_steps": (step_sum / step_count) if step_count else 0.0,
            "draw_rate": draw_count / float(sample_count),
        },
        "winner_player_id": winner_counts,
        "termination_reason": termination_reason_counts,
        "illegal_reason_distribution": illegal_reason_counts,
        "rank_top1": top1_counts,
    }


def _arena_entry(sample: Mapping[str, Any]) -> Mapping[str, Any]:
    return resolve_selected_predict_result(sample, domain="arena")


def _duration_ms(sample: Mapping[str, Any], footer: Mapping[str, Any]) -> Optional[float]:
    metadata = sample.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    header = metadata.get("game_arena")
    if not isinstance(header, Mapping):
        return None
    start_ms = _coerce_float(header.get("start_time_ms"))
    end_ms = _coerce_float(footer.get("end_time_ms"))
    if start_ms is None or end_ms is None:
        return None
    return max(0.0, end_ms - start_ms)


def _illegal_reason_distribution(entry: Mapping[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    trace = entry.get("arena_trace")
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
        return counts
    for step in trace:
        if not isinstance(step, Mapping):
            continue
        reason = step.get("illegal_reason")
        if reason in (None, ""):
            continue
        key = str(reason)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["ArenaSummaryGenerator"]
