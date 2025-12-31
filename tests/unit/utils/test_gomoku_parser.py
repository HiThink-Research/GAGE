from gage_eval.role.arena.parsers.gomoku_parser import GomokuParser


def test_parse_coord_letter_number():
    parser = GomokuParser(board_size=15)

    result = parser.parse("I will play H8")

    assert result.error is None
    assert result.coord == "H8"
    assert result.move == (7, 7)


def test_parse_coord_numeric_pair():
    parser = GomokuParser(board_size=15)

    result = parser.parse("8, 8")

    assert result.error is None
    assert result.coord == "H8"
    assert result.move == (7, 7)


def test_parse_out_of_bounds():
    parser = GomokuParser(board_size=15)

    result = parser.parse("Z9")

    assert result.error == "out_of_bounds"
    assert result.move is None


def test_parse_illegal_move_with_legal_moves():
    parser = GomokuParser(board_size=15)

    result = parser.parse("B2", legal_moves={"A1"})

    assert result.error == "illegal_move"
    assert result.coord == "B2"


def test_parse_prefers_last_coordinate():
    parser = GomokuParser(board_size=15)

    result = parser.parse("First A1 then B2")

    assert result.error is None
    assert result.coord == "B2"


def test_parse_selects_last_legal_coordinate():
    parser = GomokuParser(board_size=15)

    result = parser.parse("A1 B2", legal_moves={"B2"})

    assert result.error is None
    assert result.coord == "B2"


def test_build_rethink_prompt_includes_context():
    parser = GomokuParser(board_size=15)

    prompt = parser.build_rethink_prompt(
        last_output="My move is Z9",
        reason="out_of_bounds",
        legal_moves=["A1", "A2"],
    )

    assert "out_of_bounds" in prompt
    assert "My move is Z9" in prompt
    assert "A1" in prompt
