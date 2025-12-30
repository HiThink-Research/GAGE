from gage_eval.role.arena.parsers.gomoku_parser import GridParser


def test_tictactoe_parser_row_col() -> None:
    parser = GridParser(board_size=3, coord_scheme="ROW_COL")

    result = parser.parse("2, 3")

    assert result.error is None
    assert result.coord == "2,3"
    assert result.move == (1, 2)


def test_tictactoe_parser_out_of_bounds() -> None:
    parser = GridParser(board_size=3, coord_scheme="ROW_COL")

    result = parser.parse("4,1")

    assert result.error == "out_of_bounds"
    assert result.move is None


def test_tictactoe_parser_illegal_move_with_legal_moves() -> None:
    parser = GridParser(board_size=3, coord_scheme="ROW_COL")

    result = parser.parse("2,2", legal_moves={"1,1"})

    assert result.error == "illegal_move"
    assert result.coord == "2,2"
