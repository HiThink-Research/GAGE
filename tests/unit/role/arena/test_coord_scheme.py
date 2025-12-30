from __future__ import annotations

from gage_eval.role.arena.games.gomoku.env import GomokuLocalCore
from gage_eval.role.arena.parsers.gomoku_parser import GomokuParser


def test_coord_scheme_aa1_supports_large_board() -> None:
    core = GomokuLocalCore(board_size=28, win_len=5, coord_scheme="AA1")

    coord = core.index_to_coord(0, 26)
    row, col = core.coord_to_index(coord)

    assert coord == "AA1"
    assert (row, col) == (0, 26)


def test_coord_scheme_row_col_round_trip() -> None:
    core = GomokuLocalCore(board_size=9, win_len=5, coord_scheme="row_col")

    coord = core.index_to_coord(2, 3)
    row, col = core.coord_to_index("3,4")

    assert coord == "3,4"
    assert (row, col) == (2, 3)


def test_parser_row_col_normalizes_spaces() -> None:
    parser = GomokuParser(board_size=9, coord_scheme="row_col")

    result = parser.parse("3 4", legal_moves=["3,4", "5,6"])

    assert result.coord == "3,4"
