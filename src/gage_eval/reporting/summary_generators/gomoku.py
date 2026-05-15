"""Gomoku summary generator."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from gage_eval.evaluation.sample_envelope import resolve_selected_predict_result
from gage_eval.registry import registry
from gage_eval.reporting.contracts import SummaryGeneratorResult
from gage_eval.reporting.summary_generators import SummaryGenerator
from gage_eval.reporting.summary_generators.base import records_from_context, section


@registry.asset(
    "summary_generators",
    "gomoku_summary",
    desc="Gomoku summary generator",
    tags=("gomoku", "report"),
    default_enabled=True,
)
class GomokuSummaryGenerator(SummaryGenerator):
    """Generate aggregate summary metrics for Gomoku runs."""

    def generate(self, context: Any) -> SummaryGeneratorResult | None:
        """Build Gomoku summary statistics from cached samples.

        Args:
            context: Report context mapping with sample records.

        Returns:
            Summary result with Gomoku statistics or None if no Gomoku samples are found.
        """

        records = records_from_context(context)
        summary = _build_gomoku_summary(records)
        if not summary:
            return None
        attention_cases = []
        if summary["overall"].get("illegal_games", 0) > 0:
            first_sample_id = _first_matching_sample_id(records)
            attention_cases.append(
                {
                    "case_id": f"gomoku/{first_sample_id or 'illegal-action'}",
                    "severity": "medium",
                    "reason_codes": ["game.illegal_action"],
                    "summary": "Gomoku run contains illegal actions.",
                    "evidence_ref_ids": [],
                    "sample_id": first_sample_id,
                    "scoring": {
                        "frequency": summary["overall"]["illegal_games"] / max(1, summary["overall"]["total"]),
                        "impact": "medium",
                        "actionability": "high",
                        "priority_score": 0.63,
                    },
                }
            )
        return SummaryGeneratorResult(
            generator_id="gomoku_summary",
            summary_sections=[section("overview", "Gomoku Summary", generator_id="gomoku_summary")],
            attention_cases=attention_cases,
            legacy_payload={"gomoku_summary": summary},
        )


def _build_gomoku_summary(records: Iterable[dict[str, Any]]) -> Optional[Dict[str, Any]]:
    total = 0
    wins: Dict[str, int] = {}
    draws = 0
    illegal_games = 0
    total_moves = 0
    total_illegal_moves = 0
    result_counts: Dict[str, int] = {}

    for record in records:
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
        source = _result_source(judge_output, model_output)
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


def _first_matching_sample_id(records: Iterable[dict[str, Any]]) -> str | None:
    for record in records:
        sample = record.get("sample") if isinstance(record, dict) else None
        if isinstance(sample, dict):
            return str(sample.get("id") or record.get("sample_id") or "sample")
    return None


def _result_source(judge_output: Dict[str, Any], model_output: Dict[str, Any]) -> Dict[str, Any]:
    for payload in (judge_output, model_output):
        if not isinstance(payload, dict) or not payload:
            continue
        nested_result = payload.get("result")
        if isinstance(nested_result, dict):
            return nested_result
        return payload
    return {}


def _is_gomoku_sample(sample: Dict[str, Any], model_output: Dict[str, Any], judge_output: Dict[str, Any]) -> bool:
    for key in ("_dataset_id", "_gage_dataset_id"):
        dataset_id = sample.get(key)
        if isinstance(dataset_id, str) and "gomoku" in dataset_id.lower():
            return True

    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    if _is_gomoku_marker(metadata.get("game")):
        return True
    if _is_gomoku_marker(metadata.get("game_type")):
        return True
    game_arena = metadata.get("game_arena") if isinstance(metadata.get("game_arena"), dict) else {}
    if _is_gomoku_marker(game_arena.get("game_kit")):
        return True

    arena_entry = resolve_selected_predict_result(sample, domain="arena")
    if _payload_marks_gomoku(arena_entry):
        return True

    for payload in (judge_output, model_output):
        if _payload_marks_gomoku(payload):
            return True

    return _looks_like_legacy_gomoku_metadata(metadata)


def _payload_marks_gomoku(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    nested_sample = payload.get("sample")
    if isinstance(nested_sample, dict) and _is_gomoku_marker(nested_sample.get("game_kit")):
        return True
    header = payload.get("header")
    if isinstance(header, dict) and _is_gomoku_marker(header.get("game_kit")):
        return True
    return False


def _looks_like_legacy_gomoku_metadata(metadata: Dict[str, Any]) -> bool:
    return bool(
        metadata.get("board_size")
        and metadata.get("win_len")
        and metadata.get("coord_scheme")
        and metadata.get("win_directions")
    )


def _is_gomoku_marker(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "gomoku"


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
