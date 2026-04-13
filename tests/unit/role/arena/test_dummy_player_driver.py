from __future__ import annotations

from gage_eval.role.arena.player_drivers.dummy import DummyBoundPlayer
from gage_eval.role.arena.types import ArenaObservation


def _build_observation(*legal_moves: str) -> ArenaObservation:
    return ArenaObservation(
        board_text="board",
        legal_moves=list(legal_moves),
        active_player="enemy_bot",
    )


def test_dummy_player_skips_scripted_moves_that_are_not_currently_legal() -> None:
    player = DummyBoundPlayer(
        player_id="enemy_bot",
        display_name="enemy_bot",
        seat="enemy",
        player_kind="dummy",
        actions=("A1", "B2"),
        metadata={"driver_id": "player_driver/dummy"},
    )

    action = player.next_action(_build_observation("B2", "C3"))

    assert action.move == "B2"
    assert action.raw == "B2"


def test_dummy_player_falls_back_to_first_legal_move_when_script_is_exhausted() -> None:
    player = DummyBoundPlayer(
        player_id="enemy_bot",
        display_name="enemy_bot",
        seat="enemy",
        player_kind="dummy",
        actions=("A1",),
        metadata={"driver_id": "player_driver/dummy"},
    )

    player.next_action(_build_observation("A1", "B2"))
    action = player.next_action(_build_observation("B2", "C3"))

    assert action.move == "B2"
    assert action.raw == "B2"
    assert action.metadata["fallback"] == "first_legal"
