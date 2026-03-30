"""Tic-Tac-Toe coordinate parser owned by the GameKit."""

from __future__ import annotations

from gage_eval.game_kits.board_game.shared.grid_parser import GridCoordParser
from gage_eval.registry import registry


@registry.asset(
    "parser_impls",
    "grid_parser_v1",
    desc="Grid game move parser (configurable coordinate scheme)",
    tags=("grid", "parser"),
)
class GridParser(GridCoordParser):
    """Parse grid-game move coordinates from model output."""


__all__ = ["GridParser"]
