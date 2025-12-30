"""Gomoku-specific metrics."""

from __future__ import annotations

from typing import Any

from gage_eval.metrics.base import MetricContext, SimpleMetric
from gage_eval.metrics.utils import extract_field
from gage_eval.registry import registry
from gage_eval.role.arena.games.gomoku.env import DEFAULT_PLAYER_IDS


@registry.asset(
    "metrics",
    "gomoku_win_rate",
    desc="Win rate for a target player",
    tags=("gomoku", "game"),
    default_aggregation="mean",
)
class GomokuWinRateMetric(SimpleMetric):
    """Computes win rate for a selected player."""

    value_key = "win"

    def compute_value(self, context: MetricContext) -> tuple[float, dict]:
        """Compute the win signal for the target player.

        Args:
            context: MetricContext with model or judge outputs.

        Returns:
            A tuple of (score, metadata) for aggregation.
        """

        target_player = self.args.get("target_player")
        if not target_player:
            target_player = _resolve_default_player(context)
        target_player = str(target_player)
        prediction_field = self.args.get("prediction_field", "model_output.winner")
        winner = extract_field(context, prediction_field)
        score = 1.0 if winner == target_player else 0.0
        return score, {"winner": winner, "target_player": target_player}


def _resolve_default_player(context: MetricContext) -> str:
    sample = context.sample if isinstance(context.sample, dict) else {}
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    player_ids = metadata.get("player_ids") or []
    if isinstance(player_ids, dict):
        player_ids = list(player_ids.values())
    if isinstance(player_ids, list) and player_ids:
        return str(player_ids[0])
    return DEFAULT_PLAYER_IDS[0]


@registry.asset(
    "metrics",
    "gomoku_illegal_rate",
    desc="Illegal move rate for a game",
    tags=("gomoku", "game"),
    default_aggregation="mean",
)
class GomokuIllegalRateMetric(SimpleMetric):
    """Computes whether a game contains illegal moves."""

    value_key = "illegal"

    def compute_value(self, context: MetricContext) -> tuple[float, dict]:
        """Compute the illegal-move indicator.

        Args:
            context: MetricContext with model or judge outputs.

        Returns:
            A tuple of (score, metadata) for aggregation.
        """

        field = self.args.get("prediction_field", "model_output.illegal_move_count")
        count = extract_field(context, field, default=0) or 0
        try:
            count_val = int(count)
        except (TypeError, ValueError):
            count_val = 0
        score = 1.0 if count_val > 0 else 0.0
        return score, {"illegal_move_count": count_val}


@registry.asset(
    "metrics",
    "gomoku_avg_turns",
    desc="Average number of turns per game",
    tags=("gomoku", "game"),
    default_aggregation="mean",
)
class GomokuAverageTurnsMetric(SimpleMetric):
    """Computes average turns per game."""

    value_key = "turns"

    def compute_value(self, context: MetricContext) -> tuple[float, dict]:
        """Compute the move count for averaging.

        Args:
            context: MetricContext with model or judge outputs.

        Returns:
            A tuple of (turns, metadata) for aggregation.
        """

        field = self.args.get("prediction_field", "model_output.move_count")
        value = extract_field(context, field, default=0)
        try:
            turns = float(value)
        except (TypeError, ValueError):
            turns = 0.0
        return turns, {"move_count": turns}
