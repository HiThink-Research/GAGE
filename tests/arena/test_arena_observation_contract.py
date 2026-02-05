from gage_eval.role.arena.types import ArenaObservation


def test_arena_observation_derives_compatibility_fields_from_structured_payloads():
    obs = ArenaObservation(
        board_text="board",
        legal_moves=["a", "b"],
        active_player="player_0",
        last_move="a",
        view={"text": "view-text"},
        legal_actions={"items": ["a"]},
    )

    assert obs.view_text == "view-text"
    assert obs.last_action == "a"
    assert obs.legal_actions_items == ("a",)


def test_arena_observation_falls_back_to_legacy_fields_when_structured_missing():
    obs = ArenaObservation(
        board_text="board",
        legal_moves=["a", "b"],
        active_player="player_0",
    )

    assert obs.view_text == "board"
    assert obs.last_action is None
    assert obs.legal_actions_items == ("a", "b")

