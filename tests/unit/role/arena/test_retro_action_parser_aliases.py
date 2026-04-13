from gage_eval.game_kits.real_time_game.retro_platformer.parser import RetroActionParser


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


def test_retro_action_parser_handles_empty_and_invalid_hold_ticks():
    parser = RetroActionParser(hold_ticks_min=1, hold_ticks_max=12, default_hold_ticks=6)

    empty = parser.parse("", legal_moves=["noop"])
    assert empty.error == "empty_text"

    invalid_hold = parser.parse('{"move":"noop","hold_ticks":"abc"}', legal_moves=["noop"])
    assert invalid_hold.error == "invalid_hold_ticks"


def test_retro_action_parser_build_action_dict_includes_hold_ticks():
    parser = RetroActionParser(hold_ticks_min=1, hold_ticks_max=12, default_hold_ticks=6)
    parsed = parser.parse('{"move":"noop","hold_ticks":4}', legal_moves=["noop"])

    payload = parser.build_action_dict(player="player_0", parse_result=parsed)
    assert payload == {"player": "player_0", "move": "noop", "raw": '{"move":"noop","hold_ticks":4}', "hold_ticks": 4}


def test_retro_action_parser_build_rethink_prompt_keeps_json_schema_literal():
    parser = RetroActionParser(hold_ticks_min=1, hold_ticks_max=12, default_hold_ticks=6)

    prompt = parser.build_rethink_prompt(
        last_output="bad output",
        reason="illegal_move",
        legal_moves=["noop", "right"],
    )

    assert '{"move": "<legal_move_or_key_combo>", "hold_ticks": <int>}' in prompt
    assert "Legal moves: noop, right." in prompt
