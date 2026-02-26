from gage_eval.role.arena.games.retro.retro_input_mapper import (
    RetroInputMapper,
    _coerce_bool,
    _normalize_key,
    _resolve_hold_ticks,
)


def test_retro_input_mapper_emits_actions_and_dedups_same_move():
    mapper = RetroInputMapper(default_hold_ticks=3, dedup_same_move=True)

    first = mapper.handle_browser_event(
        {"type": "keydown", "key": "d"},
        context={"human_player_id": "player_9"},
    )
    assert len(first) == 1
    assert first[0].player_id == "player_9"
    assert first[0].move == "RIGHT"
    assert first[0].metadata["hold_ticks"] == 3

    second = mapper.handle_browser_event(
        {"type": "keydown", "key": "d"},
        context={"human_player_id": "player_9"},
    )
    assert second == []


def test_retro_input_mapper_handles_keys_state_and_keyup():
    mapper = RetroInputMapper(default_hold_ticks=2, dedup_same_move=False)

    combo = mapper.handle_browser_event(
        {"type": "keys_state", "keys": {"d": True, "k": True}},
        context={"human_player_id": "p0"},
    )
    assert len(combo) == 1
    assert combo[0].move == "RIGHT+B"

    release = mapper.handle_browser_event(
        {"type": "keyup", "key": "d"},
        context={"human_player_id": "p0"},
    )
    assert len(release) == 1
    assert release[0].move == "B"


def test_retro_input_mapper_reads_hold_ticks_from_alias_field():
    mapper = RetroInputMapper(default_hold_ticks=1, dedup_same_move=False)
    actions = mapper.handle_browser_event(
        {"event_type": "custom", "key": "j", "pressed": True, "holdTicks": 5},
        context={"human_player_id": "p1"},
    )

    assert len(actions) == 1
    assert actions[0].move == "A"
    assert actions[0].metadata["hold_ticks"] == 5


def test_retro_input_mapper_helper_functions():
    assert _normalize_key("  ArrowUp ") == "arrowup"
    assert _coerce_bool("on", default=False) is True
    assert _coerce_bool("off", default=True) is False
    assert _coerce_bool("invalid", default=True) is True
    assert _resolve_hold_ticks({"hold_ticks": "7"}, default=2) == 7
    assert _resolve_hold_ticks({"holdTicks": "not-int"}, default=2) == 2
