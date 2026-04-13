"""Gomoku-specific coordinate parser owned by the GameKit."""

from __future__ import annotations

from gage_eval.game_kits.board_game.shared.grid_parser import (
    GridCoordParseResult,
    GridCoordParser,
)
from gage_eval.registry import registry

GomokuParseResult = GridCoordParseResult


@registry.asset(
    "parser_impls",
    "gomoku_v1",
    desc="Gomoku move parser (configurable coordinate scheme)",
    tags=("gomoku", "parser"),
)
class GomokuParser(GridCoordParser):
    """Parse Gomoku move coordinates from model output."""


__all__ = ["GomokuParseResult", "GomokuParser"]
