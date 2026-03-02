from __future__ import annotations

from gage_eval.role.arena.types import ArenaFooter, ArenaHeader, ArenaObservation, ArenaTraceStep


def test_arena_observation_property_fallbacks() -> None:
    obs = ArenaObservation(
        board_text="board",
        legal_moves=["left", "right"],
        active_player="p0",
        metadata={},
    )
    assert obs.view_text == "board"
    assert obs.legal_actions_items == ["left", "right"]
    assert obs.last_action is None

    obs2 = ArenaObservation(
        board_text="board2",
        legal_moves=["m1"],
        active_player="p1",
        last_move="m0",
        metadata={"last_move": "ignored"},
        view={"text": None},
        legal_actions={"items": [1, "m2"]},
    )
    assert obs2.view_text == "board2"
    assert obs2.legal_actions_items == ["1", "m2"]
    assert obs2.last_action == "m0"

    obs3 = ArenaObservation(
        board_text="board3",
        legal_moves=["m1"],
        active_player="p2",
        metadata={"last_move": 7},
    )
    assert obs3.last_action == "7"


def test_arena_contract_dataclass_to_dict() -> None:
    trace = ArenaTraceStep(
        step_index=1,
        trace_state="done",
        timestamp=100,
        player_id="p0",
        action_raw="a",
        action_applied="a",
        t_obs_ready_ms=100,
        t_action_submitted_ms=101,
        timeout=False,
        is_action_legal=True,
        retry_count=0,
        info={"k": "v"},
    )
    footer = ArenaFooter(
        end_time_ms=200,
        total_steps=1,
        winner_player_id="p0",
        termination_reason="finished",
        ranks=["p0"],
        final_scores={"p0": 1.0},
        episode_returns={"p0": 1.0},
    )
    header = ArenaHeader(
        engine_id="env",
        seed=1,
        mode="single",
        players=[{"player_id": "p0", "controller_type": "agent", "model_id": None, "policy_id": None}],
        start_time_ms=0,
    )

    assert trace.to_dict()["player_id"] == "p0"
    assert footer.to_dict()["termination_reason"] == "finished"
    assert header.to_dict()["engine_id"] == "env"
