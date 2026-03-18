from __future__ import annotations

from typing import Optional

from gage_eval.role.arena.schedulers._scheduler_utils import (
    make_trace_entry,
    set_trace_action_fields,
    think_with_timeout,
)
from gage_eval.role.arena.types import ArenaAction, ArenaObservation


def _build_observation() -> ArenaObservation:
    return ArenaObservation(
        board_text="",
        legal_moves=["0", "1"],
        active_player="p0",
    )


class _AsyncPlayerReady:
    def __init__(self) -> None:
        self.started = False
        self.deadline_ms: Optional[int] = None
        self._poll_count = 0
        self._action = ArenaAction(player="p0", move="1", raw="1")

    def think(self, observation: ArenaObservation) -> ArenaAction:
        raise AssertionError("think() should not be called for async-capable players")

    def start_thinking(self, observation: ArenaObservation, *, deadline_ms: Optional[int] = None) -> bool:
        _ = observation
        self.started = True
        self.deadline_ms = deadline_ms
        return True

    def has_action(self) -> bool:
        self._poll_count += 1
        return self._poll_count >= 2

    def pop_action(self) -> ArenaAction:
        return self._action


class _AsyncPlayerNeverReady:
    def start_thinking(self, observation: ArenaObservation, *, deadline_ms: Optional[int] = None) -> bool:
        _ = observation, deadline_ms
        return True

    def has_action(self) -> bool:
        return False

    def pop_action(self) -> ArenaAction:
        raise AssertionError("pop_action() should not be called when no action is ready")


class _SyncPlayer:
    def __init__(self) -> None:
        self.called = False

    def think(self, observation: ArenaObservation) -> ArenaAction:
        _ = observation
        self.called = True
        return ArenaAction(player="p0", move="0", raw="0")


def test_think_with_timeout_prefers_async_player_api() -> None:
    player = _AsyncPlayerReady()

    action, timed_out, error_type = think_with_timeout(
        player=player,
        observation=_build_observation(),
        timeout_ms=50,
    )

    assert action is not None
    assert action.move == "1"
    assert timed_out is False
    assert error_type is None
    assert player.started is True
    assert player.deadline_ms == 50


def test_think_with_timeout_async_path_respects_timeout() -> None:
    action, timed_out, error_type = think_with_timeout(
        player=_AsyncPlayerNeverReady(),
        observation=_build_observation(),
        timeout_ms=1,
    )

    assert action is None
    assert timed_out is True
    assert error_type == "timeout"


def test_think_with_timeout_sync_player_without_timeout() -> None:
    player = _SyncPlayer()

    action, timed_out, error_type = think_with_timeout(
        player=player,
        observation=_build_observation(),
        timeout_ms=None,
    )

    assert action is not None
    assert action.move == "0"
    assert timed_out is False
    assert error_type is None
    assert player.called is True


def test_set_trace_action_fields_prefers_semantic_applied_value() -> None:
    entry = make_trace_entry(
        step_index=0,
        player_id="p0",
        timestamp_ms=1,
        t_obs_ready_ms=2,
    )
    action = ArenaAction(
        player="p0",
        move="1",
        raw="Action: 1",
        metadata={"trace_action_applied": "ATTACK", "player_type": "backend"},
    )

    set_trace_action_fields(entry, action, action_format="flat")

    assert entry["action_raw"] == "Action: 1"
    assert entry["action_applied"] == "ATTACK"


def test_set_trace_action_fields_strips_internal_trace_metadata_from_envelope() -> None:
    entry = make_trace_entry(
        step_index=0,
        player_id="p0",
        timestamp_ms=1,
        t_obs_ready_ms=2,
    )
    action = ArenaAction(
        player="p0",
        move="1",
        raw="Action: 1",
        metadata={"trace_action_applied": "ATTACK", "player_type": "backend"},
    )

    set_trace_action_fields(entry, action, action_format="envelope")

    assert entry["action_applied"]["move"] == "ATTACK"
    assert entry["action_applied"]["metadata"] == {"player_type": "backend"}
    assert entry["action_raw"]["metadata"] == {"player_type": "backend"}
