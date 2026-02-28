import pytest

from gage_eval.role.arena.games.retro.keyboard_input import (
    KeyState,
    build_default_key_map,
    resolve_macro_move_from_action_state,
)


def test_default_key_map_matches_wasd_jkl_ground_truth():
    key_map = build_default_key_map()

    assert key_map["w"] == "up"
    assert key_map["a"] == "left"
    assert key_map["s"] == "down"
    assert key_map["d"] == "right"

    assert key_map["j"] == "jump"
    assert key_map["k"] == "run"
    assert key_map["l"] == "select"

    assert key_map["Enter"] == "start"


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"start": True}, "start"),
        ({"select": True}, "select"),
        ({"left": True, "right": True}, "noop"),
        ({"right": True}, "right"),
        ({"right": True, "run": True}, "right_run"),
        ({"right": True, "jump": True}, "right_jump"),
        ({"right": True, "run": True, "jump": True}, "right_run_jump"),
        ({"left": True, "run": True}, "left_run"),
        ({"up": True}, "up"),
        ({"down": True}, "down"),
        ({"jump": True}, "jump"),
        ({"run": True}, "run"),
        ({}, "noop"),
    ],
)
def test_resolve_macro_move_from_action_state(state, expected):
    assert resolve_macro_move_from_action_state(state) == expected


def test_key_state_normalizes_keys_and_respects_legal_moves():
    key_state = KeyState(build_default_key_map(), legal_moves=["noop", "right_run_jump"])

    key_state.set_key("d", True)
    key_state.set_key("k", True)
    key_state.set_key("j", True)
    assert key_state.resolve_move() == "right_run_jump"

    # Arrow keys can be used instead of WASD.
    arrows = KeyState(build_default_key_map(), legal_moves=["noop", "right_run_jump"])
    arrows.set_key("ArrowRight", True)
    arrows.set_key("k", True)
    arrows.set_key("j", True)
    assert arrows.resolve_move() == "right_run_jump"

    # Unknown keys are ignored and do not crash.
    key_state.set_key("?", True)
    assert key_state.resolve_move() == "right_run_jump"

    # Moves outside the allowed set become noop.
    key_state.set_key("j", False)
    key_state.set_key("k", False)
    assert key_state.resolve_move() == "noop"


def test_key_state_update_from_payload_updates_action_state_snapshot():
    key_state = KeyState(build_default_key_map())
    key_state.update_from_payload({"D": True, "k": True, "j": True, "?": True})

    snapshot = key_state.snapshot()
    assert snapshot["right"] is True
    assert snapshot["run"] is True
    assert snapshot["jump"] is True
