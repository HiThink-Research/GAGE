"""Gomoku summary generator."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.evaluation.cache import EvalCache
from gage_eval.registry import registry
from gage_eval.reporting.summary_generators import SummaryGenerator


@registry.asset(
    "summary_generators",
    "gomoku_summary",
    desc="Gomoku summary generator",
    tags=("gomoku", "report"),
    default_enabled=True,
)
class GomokuSummaryGenerator(SummaryGenerator):
    """Generate aggregate summary metrics for Gomoku runs."""

    def generate(self, cache: EvalCache) -> Optional[Dict[str, Any]]:
        """Build Gomoku summary statistics from cached samples.

        Args:
            cache: EvalCache storing sample payloads.

        Returns:
            Summary payload with Gomoku statistics or None if no Gomoku samples are found.
        """

        summary = _build_gomoku_summary(cache)
        if not summary:
            return None
        return {"gomoku_summary": summary}


def _build_gomoku_summary(cache: EvalCache) -> Optional[Dict[str, Any]]:
    total = 0
    wins: Dict[str, int] = {}
    draws = 0
    illegal_games = 0
    total_moves = 0
    total_illegal_moves = 0
    result_counts: Dict[str, int] = {}

    for record in cache.iter_samples():
        if not isinstance(record, dict):
            continue
        sample = record.get("sample") if isinstance(record.get("sample"), dict) else None
        if not sample:
            continue
        model_output = record.get("model_output") if isinstance(record.get("model_output"), dict) else {}
        judge_output = record.get("judge_output") if isinstance(record.get("judge_output"), dict) else {}

        if not _is_gomoku_sample(sample, model_output, judge_output):
            continue

        total += 1
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        for player_id in _extract_player_ids(metadata):
            wins.setdefault(player_id, 0)
        source = judge_output or model_output
        winner = source.get("winner")
        result = source.get("result")
        move_count = _to_int(source.get("move_count"))
        illegal_count = _to_int(source.get("illegal_move_count"))

        total_moves += move_count
        total_illegal_moves += illegal_count
        if illegal_count > 0:
            illegal_games += 1

        if winner:
            winner_key = str(winner)
            wins[winner_key] = wins.get(winner_key, 0) + 1

        if result is None and winner in (None, "", 0):
            result = "draw"
        if result:
            result_key = str(result)
            result_counts[result_key] = result_counts.get(result_key, 0) + 1
            if result_key == "draw":
                draws += 1

    if total == 0:
        return None

    return {
        "overall": {
            "total": total,
            "draws": draws,
            "illegal_games": illegal_games,
            "avg_moves": total_moves / total if total else 0.0,
            "avg_illegal_moves": total_illegal_moves / total if total else 0.0,
        },
        "wins": wins,
        "results": result_counts,
    }


def _is_gomoku_sample(sample: Dict[str, Any], model_output: Dict[str, Any], judge_output: Dict[str, Any]) -> bool:
    dataset_id = sample.get("_dataset_id")
    if isinstance(dataset_id, str) and "gomoku" in dataset_id.lower():
        return True

    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    if metadata.get("game") == "gomoku":
        return True
    if metadata.get("board_size") and metadata.get("win_len") and metadata.get("coord_scheme"):
        return True

    for payload in (judge_output, model_output):
        if payload.get("game_log") or payload.get("move_count"):
            return True

    return False


def _extract_player_ids(metadata: Dict[str, Any]) -> list[str]:
    player_ids = metadata.get("player_ids")
    if isinstance(player_ids, dict):
        player_ids = list(player_ids.values())
    if isinstance(player_ids, list):
        return [str(player_id) for player_id in player_ids if player_id]
    player_names = metadata.get("player_names")
    if isinstance(player_names, dict):
        return [str(player_id) for player_id in player_names.keys() if player_id]
    return []


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["GomokuSummaryGenerator"]
