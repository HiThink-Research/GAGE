from gage_eval.role.arena.parsers.retro_action_parser import RetroActionParser


def test_retro_action_parser_accepts_key_combo_aliases():
    parser = RetroActionParser(hold_ticks_min=1, hold_ticks_max=12, default_hold_ticks=6)

    result = parser.parse('{"move":"d+j+k","hold_ticks":5}', legal_moves=["noop", "right_run_jump"])

    assert result.error is None
    assert result.move == "right_run_jump"
    assert result.hold_ticks == 5


def test_retro_action_parser_accepts_compact_key_aliases():
    parser = RetroActionParser(hold_ticks_min=1, hold_ticks_max=12, default_hold_ticks=6)

    result = parser.parse("dk", legal_moves=["noop", "right_run"])

    assert result.error is None
    assert result.move == "right_run"
    assert result.hold_ticks == 6


def test_retro_action_parser_accepts_select_key_alias():
    parser = RetroActionParser(hold_ticks_min=1, hold_ticks_max=12, default_hold_ticks=6)

    result = parser.parse("l", legal_moves=["noop", "select"])

    assert result.error is None
    assert result.move == "select"


def test_retro_action_parser_rejects_key_alias_when_illegal():
    parser = RetroActionParser(hold_ticks_min=1, hold_ticks_max=12, default_hold_ticks=6)

    result = parser.parse("dk", legal_moves=["noop", "right"])

    assert result.error == "illegal_move"
    assert result.move == "right_run"
