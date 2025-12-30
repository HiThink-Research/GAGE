from __future__ import annotations

from gage_eval.role.arena.games.gomoku.rules import GomokuRuleEngine


def _empty_board(size: int) -> list[list[str]]:
    return [["." for _ in range(size)] for _ in range(size)]


def test_rule_profile_freestyle_allows_overline() -> None:
    board = _empty_board(6)
    for col in range(6):
        board[0][col] = "B"

    engine = GomokuRuleEngine(win_len=5, rule_profile="freestyle")

    info = engine.find_win_info(board, 0, 5)

    assert info is not None
    assert info.line_length == 6
    assert info.direction == "horizontal"


def test_rule_profile_exact_disallows_overline() -> None:
    board = _empty_board(6)
    for col in range(6):
        board[0][col] = "B"

    engine = GomokuRuleEngine(win_len=5, rule_profile="exact")

    assert engine.find_win_info(board, 0, 5) is None


def test_win_directions_restrict_detection() -> None:
    board = _empty_board(5)
    for row in range(5):
        board[row][0] = "W"

    horizontal_only = GomokuRuleEngine(win_len=5, win_directions=["horizontal"])
    assert horizontal_only.find_win_info(board, 4, 0) is None

    vertical_only = GomokuRuleEngine(win_len=5, win_directions=["vertical"])
    info = vertical_only.find_win_info(board, 4, 0)
    assert info is not None
    assert info.direction == "vertical"
