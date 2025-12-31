from __future__ import annotations

from gage_eval.registry import registry
from gage_eval.role.arena.parsers.gomoku_parser import GomokuParser, GridParser


def test_parser_registry_resolves_gomoku() -> None:
    parser_cls = registry.get("parser_impls", "gomoku_v1")

    assert parser_cls is GomokuParser

    parser = parser_cls(board_size=5)
    result = parser.parse("A1", legal_moves=["A1", "B2"])

    assert result.coord == "A1"


def test_parser_registry_resolves_grid_parser() -> None:
    parser_cls = registry.get("parser_impls", "grid_parser_v1")

    assert parser_cls is GridParser

    parser = parser_cls(board_size=3, coord_scheme="ROW_COL")
    result = parser.parse("2,2", legal_moves=["2,2", "3,3"])

    assert result.coord == "2,2"
