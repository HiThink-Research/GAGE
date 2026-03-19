from __future__ import annotations

from gage_eval.registry import registry
from gage_eval.role.arena.parsers.gomoku_parser import GomokuParser, GridParser
from gage_eval.role.arena.parsers.vizdoom_parser import VizDoomParser


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


def test_parser_registry_resolves_vizdoom_parser() -> None:
    parser_cls = registry.get("parser_impls", "vizdoom_parser_v1")

    assert parser_cls is VizDoomParser

    parser = parser_cls(default_action=0)
    result = parser.parse('{"action": 2}', legal_moves=["1", "2", "3"])

    assert result.coord == "2"
    assert result.error is None


def test_vizdoom_parser_parses_action_with_reason_lines() -> None:
    parser = VizDoomParser(default_action=0)

    result = parser.parse(
        "Action: 2\nReason: Enemy appears on the left side, so move left to center.",
        legal_moves=["1", "2", "3"],
    )

    assert result.coord == "2"
    assert result.error is None
    assert result.reason == "Enemy appears on the left side, so move left to center."


def test_vizdoom_parser_parses_json_reason_payload() -> None:
    parser = VizDoomParser(default_action=0)

    result = parser.parse(
        '{"action": 3, "reason": "Enemy is right of center; adjust right before firing."}',
        legal_moves=["1", "2", "3"],
    )

    assert result.coord == "3"
    assert result.error is None
    assert result.reason == "Enemy is right of center; adjust right before firing."
