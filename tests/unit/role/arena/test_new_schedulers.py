from __future__ import annotations

import time
from typing import Any, Optional

import pytest

from gage_eval.role.adapters.arena import ArenaRoleAdapter
from gage_eval.role.arena.schedulers.multi_timeline_scheduler import MultiTimelineScheduler
from gage_eval.role.arena.schedulers.record_scheduler import RecordScheduler
from gage_eval.role.arena.schedulers.simultaneous_scheduler import SimultaneousScheduler
from gage_eval.role.arena.schedulers.tick_scheduler import TickScheduler
from gage_eval.role.arena.types import ArenaAction, ArenaObservation, GameResult


class _StaticPlayer:
    def __init__(self, name: str, move: str, *, delay_s: float = 0.0) -> None:
        self.name = name
        self._move = move
        self._delay_s = max(0.0, float(delay_s))

    def think(self, observation: ArenaObservation) -> ArenaAction:
        if self._delay_s > 0:
            time.sleep(self._delay_s)
        return ArenaAction(player=self.name, move=self._move, raw=self._move, metadata={})


class _SingleActionEnv:
    def __init__(self) -> None:
        self._terminal = False
        self.applied_actions: list[Any] = []

    def reset(self) -> None:
        self._terminal = False
        self.applied_actions = []

    def get_active_player(self) -> str:
        return "p0"

    def observe(self, player: str) -> ArenaObservation:
        return ArenaObservation(
            board_text="",
            legal_moves=["UP", "NOOP"],
            active_player=str(player),
            metadata={},
        )

    def apply(self, action: Any) -> Optional[GameResult]:
        self.applied_actions.append(action)
        self._terminal = True
        return self.build_result(result="draw", reason="terminated")

    def is_terminal(self) -> bool:
        return self._terminal

    def build_result(self, *, result: str, reason: Optional[str]) -> GameResult:
        return GameResult(
            winner=None,
            result=result,
            reason=reason,
            move_count=len(self.applied_actions),
            illegal_move_count=0,
            final_board="",
            move_log=[],
        )


class _TwoTickDictEnv:
    def __init__(self, *, max_ticks: int = 2) -> None:
        self._max_ticks = max(1, int(max_ticks))
        self._tick_count = 0
        self.applied_actions: list[Any] = []

    def reset(self) -> None:
        self._tick_count = 0
        self.applied_actions = []

    def get_active_player(self) -> str:
        return "p0"

    def observe(self, player: str) -> ArenaObservation:
        return ArenaObservation(
            board_text="",
            legal_moves=["A", "B", "NOOP"],
            active_player=str(player),
            metadata={},
        )

    def apply(self, action: Any) -> Optional[GameResult]:
        self.applied_actions.append(action)
        self._tick_count += 1
        if self._tick_count >= self._max_ticks:
            return self.build_result(result="draw", reason="terminated")
        return None

    def is_terminal(self) -> bool:
        return self._tick_count >= self._max_ticks

    def build_result(self, *, result: str, reason: Optional[str]) -> GameResult:
        return GameResult(
            winner=None,
            result=result,
            reason=reason,
            move_count=self._tick_count,
            illegal_move_count=0,
            final_board="",
            move_log=[],
        )


class _AsyncTickPlayer:
    def __init__(self, name: str, moves: list[str]) -> None:
        self.name = name
        self._moves = list(moves)
        self._inflight = False
        self._ready: list[ArenaAction] = []
        self.start_calls = 0

    def start_thinking(self, observation: ArenaObservation, *, deadline_ms: Optional[int] = None) -> bool:
        _ = observation, deadline_ms
        if self._inflight:
            return False
        self._inflight = True
        self.start_calls += 1
        move = self._moves.pop(0) if self._moves else "NOOP"
        self._ready.append(ArenaAction(player=self.name, move=move, raw=move, metadata={}))
        return True

    def has_action(self) -> bool:
        return bool(self._ready)

    def pop_action(self) -> ArenaAction:
        action = self._ready.pop(0)
        self._inflight = False
        return action


class _QueueReadyAsyncPlayer:
    def __init__(self, name: str, moves: list[str]) -> None:
        self.name = name
        self._moves = list(moves)
        self._inflight = False
        self._ready: list[ArenaAction] = []
        self.start_calls = 0
        self.started_while_ready = 0

    def start_thinking(self, observation: ArenaObservation, *, deadline_ms: Optional[int] = None) -> bool:
        _ = observation, deadline_ms
        if self._inflight:
            return False
        if self._ready:
            self.started_while_ready += 1
        self._inflight = True
        self.start_calls += 1
        move = self._moves.pop(0) if self._moves else "NOOP"
        # Emulate async completion happening before the scheduler pops current action.
        self._ready.append(ArenaAction(player=self.name, move=move, raw=move, metadata={}))
        self._inflight = False
        return True

    def has_action(self) -> bool:
        return bool(self._ready)

    def pop_action(self) -> ArenaAction:
        return self._ready.pop(0)


def test_record_scheduler_timeout_fallback_and_trace() -> None:
    environment = _SingleActionEnv()
    player = _StaticPlayer("p0", "UP", delay_s=0.05)
    scheduler = RecordScheduler(
        tick_ms=0,
        action_timeout_ms=5,
        timeout_fallback_move="NOOP",
    )

    result = scheduler.run_loop(environment, [player])

    assert environment.applied_actions
    submitted = environment.applied_actions[0]
    assert isinstance(submitted, ArenaAction)
    assert submitted.move == "NOOP"
    assert result.arena_trace
    trace = result.arena_trace[0]
    assert trace["timeout"] is True
    assert trace["trace_state"] == "done"
    assert trace["player_id"] == "p0"
    assert trace["action_applied"] == "NOOP"


def test_record_scheduler_trace_options_are_configurable() -> None:
    environment = _SingleActionEnv()
    player = _StaticPlayer("p0", "UP")
    scheduler = RecordScheduler(
        tick_ms=0,
        max_steps=1,
        trace_step_index_start=1,
        trace_timestamp_clock="monotonic",
        trace_time_clock="wall_clock",
        trace_finalize_timing="after_action_submit",
        trace_action_format="envelope",
    )

    result = scheduler.run_loop(environment, [player])

    assert len(result.arena_trace) == 1
    trace = result.arena_trace[0]
    assert trace["step_index"] == 1
    assert trace["trace_state"] == "done"
    assert isinstance(trace["action_raw"], dict)
    assert isinstance(trace["action_applied"], dict)
    assert trace["action_raw"]["raw"] == "UP"
    assert trace["action_applied"]["move"] == "UP"
    assert trace["t_obs_ready_ms"] - trace["timestamp"] > 1_000_000


def test_simultaneous_scheduler_collects_multi_player_actions() -> None:
    environment = _SingleActionEnv()
    players = [
        _StaticPlayer("p0", "UP"),
        _StaticPlayer("p1", "DOWN", delay_s=0.05),
    ]
    scheduler = SimultaneousScheduler(
        frames_per_action=1,
        max_steps=1,
        action_timeout_ms=5,
        timeout_fallback_move="NOOP",
        tick_ms=0,
        trace_step_index_start=3,
        trace_action_format="envelope",
    )

    result = scheduler.run_loop(environment, players)

    assert environment.applied_actions
    action_map = environment.applied_actions[0]
    assert isinstance(action_map, dict)
    assert action_map["p0"].move == "UP"
    assert action_map["p1"].move == "NOOP"
    assert len(result.arena_trace) == 2
    assert result.arena_trace[0]["step_index"] == 3
    assert result.arena_trace[1]["step_index"] == 4
    assert isinstance(result.arena_trace[0]["action_raw"], dict)
    assert {entry["player_id"] for entry in result.arena_trace} == {"p0", "p1"}


def test_multi_timeline_scheduler_applies_once_per_tick_and_writes_timeline_id() -> None:
    environment = _TwoTickDictEnv(max_ticks=2)
    players = [
        _StaticPlayer("p0", "L"),
        _StaticPlayer("p1", "R"),
    ]
    scheduler = MultiTimelineScheduler(
        tick_ms=1,
        max_ticks=2,
        default_fallback_move="NOOP",
        trace_step_index_start=10,
        trace_action_format="envelope",
        lane_registry={
            "lane_record_human": {"type": "record", "action_timeout_ms": 50},
            "lane_sim_ai": {"type": "simultaneous", "action_timeout_ms": 50},
        },
        timelines=[
            {"timeline_id": "timeline_human", "player_ids": ["p0"], "lane_ref": "lane_record_human"},
            {"timeline_id": "timeline_ai", "player_ids": ["p1"], "lane_ref": "lane_sim_ai"},
        ],
    )

    result = scheduler.run_loop(environment, players)

    assert len(environment.applied_actions) == 2
    assert all(isinstance(action, dict) for action in environment.applied_actions)
    assert all(set(action.keys()) == {"p0", "p1"} for action in environment.applied_actions)
    assert len(result.arena_trace) == 4
    assert [entry["step_index"] for entry in result.arena_trace] == [10, 11, 12, 13]
    assert isinstance(result.arena_trace[0]["action_applied"], dict)
    assert {entry.get("timeline_id") for entry in result.arena_trace} == {
        "timeline_human",
        "timeline_ai",
    }


def test_multi_timeline_scheduler_classifies_lane_types_and_completes_action_map() -> None:
    environment = _SingleActionEnv()
    players = [
        _StaticPlayer("p0", "L"),
        _StaticPlayer("p1", "R"),
        _StaticPlayer("p2", "U"),
    ]
    scheduler = MultiTimelineScheduler(
        tick_ms=1,
        max_ticks=1,
        default_fallback_move="NOOP",
        lane_registry={
            "lane_record": {"type": "record", "action_timeout_ms": 50, "timeout_fallback_move": "NOOP"},
            "lane_sim": {"type": "simultaneous", "action_timeout_ms": 50, "timeout_fallback_move": "NOOP"},
        },
        timelines=[
            {"timeline_id": "timeline_record", "player_ids": ["p0", "p1"], "lane_ref": "lane_record"},
            {"timeline_id": "timeline_sim", "player_ids": ["p2"], "lane_ref": "lane_sim"},
        ],
    )

    result = scheduler.run_loop(environment, players)

    assert environment.applied_actions
    action_map = environment.applied_actions[0]
    assert isinstance(action_map, dict)
    assert set(action_map.keys()) == {"p0", "p1", "p2"}
    assert action_map["p0"].move == "L"
    assert action_map["p2"].move == "U"
    assert action_map["p1"].move == "NOOP"
    assert action_map["p1"].metadata.get("fallback") == "missing_lane_action"
    assert {entry["player_id"] for entry in result.arena_trace} == {"p0", "p2"}


def test_multi_timeline_scheduler_tick_lane_collects_all_lane_players() -> None:
    environment = _SingleActionEnv()
    players = [
        _StaticPlayer("p0", "UP"),
        _StaticPlayer("p1", "DOWN"),
    ]
    scheduler = MultiTimelineScheduler(
        tick_ms=1,
        max_ticks=1,
        default_fallback_move="NOOP",
        lane_registry={
            "lane_tick": {"type": "tick", "action_timeout_ms": 50, "timeout_fallback_move": "NOOP"},
        },
        timelines=[
            {"timeline_id": "timeline_tick", "player_ids": ["p0", "p1"], "lane_ref": "lane_tick"},
        ],
    )

    result = scheduler.run_loop(environment, players)

    assert environment.applied_actions
    action_map = environment.applied_actions[0]
    assert isinstance(action_map, dict)
    assert set(action_map.keys()) == {"p0", "p1"}
    assert action_map["p0"].move == "UP"
    assert action_map["p1"].move == "DOWN"
    assert action_map["p0"].metadata.get("fallback") is None
    assert action_map["p1"].metadata.get("fallback") is None
    assert {entry["player_id"] for entry in result.arena_trace} == {"p0", "p1"}


def test_arena_adapter_build_scheduler_supports_new_types() -> None:
    sample = {"eval_config": {"max_turns": 10}}
    record_adapter = ArenaRoleAdapter(adapter_id="arena", scheduler={"type": "record"})
    simultaneous_adapter = ArenaRoleAdapter(adapter_id="arena", scheduler={"type": "simultaneous"})
    timeline_adapter = ArenaRoleAdapter(
        adapter_id="arena",
        scheduler={
            "type": "multi_timeline",
            "lane_registry": {"lane_record": {"type": "record"}},
            "timelines": [{"timeline_id": "t0", "player_ids": ["p0"], "lane_ref": "lane_record"}],
        },
    )

    assert isinstance(record_adapter._build_scheduler(sample), RecordScheduler)
    assert isinstance(simultaneous_adapter._build_scheduler(sample), SimultaneousScheduler)
    assert isinstance(timeline_adapter._build_scheduler(sample), MultiTimelineScheduler)


def test_tick_scheduler_rearms_async_player_after_each_action() -> None:
    environment = _TwoTickDictEnv(max_ticks=3)
    player = _AsyncTickPlayer("p0", ["A", "B", "C"])
    scheduler = TickScheduler(tick_ms=1, max_ticks=10)

    result = scheduler.run_loop(environment, [player])

    assert [action.move for action in environment.applied_actions] == ["A", "B", "C"]
    assert player.start_calls >= 3
    assert result.move_count == 3
    assert len(result.arena_trace) == 3
    assert [entry["action_applied"] for entry in result.arena_trace] == ["A", "B", "C"]
    assert all(entry["trace_state"] == "done" for entry in result.arena_trace)


def test_tick_scheduler_does_not_start_new_think_when_action_is_ready() -> None:
    environment = _SingleActionEnv()
    player = _QueueReadyAsyncPlayer("p0", ["A", "B"])
    scheduler = TickScheduler(tick_ms=1, max_ticks=10)

    result = scheduler.run_loop(environment, [player])

    assert [action.move for action in environment.applied_actions] == ["A"]
    assert player.start_calls == 1
    assert player.started_while_ready == 0
    assert result.move_count == 1
    assert len(result.arena_trace) == 1
    assert result.arena_trace[0]["player_id"] == "p0"
    assert result.arena_trace[0]["action_applied"] == "A"


def test_arena_adapter_build_scheduler_applies_trace_options() -> None:
    sample = {"eval_config": {"max_turns": 10}}
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        scheduler={
            "type": "record",
            "trace_step_index_start": 8,
            "trace_timestamp_clock": "wall_clock",
            "trace_time_clock": "wall_clock",
            "trace_finalize_timing": "after_action_submit",
            "trace_action_format": "flat",
            "trace": {
                "step_index_start": 2,
                "timestamp_clock": "monotonic",
                "time_clock": "monotonic",
            },
        },
    )

    scheduler = adapter._build_scheduler(sample)

    assert isinstance(scheduler, RecordScheduler)
    assert scheduler._trace_step_index_start == 2
    assert scheduler._trace_timestamp_clock == "monotonic"
    assert scheduler._trace_time_clock == "monotonic"
    assert scheduler._trace_finalize_timing == "after_action_submit"
    assert scheduler._trace_action_format == "flat"


def test_arena_adapter_build_tick_scheduler_applies_trace_options() -> None:
    sample = {"eval_config": {"max_turns": 10}}
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        scheduler={
            "type": "tick",
            "tick_ms": 5,
            "trace_step_index_start": 6,
            "trace_timestamp_clock": "wall_clock",
            "trace_time_clock": "wall_clock",
            "trace_finalize_timing": "after_action_submit",
            "trace_action_format": "envelope",
            "trace": {
                "step_index_start": 9,
                "timestamp_clock": "monotonic",
                "time_clock": "monotonic",
            },
        },
    )

    scheduler = adapter._build_scheduler(sample)

    assert isinstance(scheduler, TickScheduler)
    assert scheduler._trace_step_index_start == 9
    assert scheduler._trace_timestamp_clock == "monotonic"
    assert scheduler._trace_time_clock == "monotonic"
    assert scheduler._trace_finalize_timing == "after_action_submit"
    assert scheduler._trace_action_format == "envelope"


def test_arena_adapter_build_scheduler_unknown_type_raises() -> None:
    adapter = ArenaRoleAdapter(adapter_id="arena", scheduler={"type": "unknown"})
    with pytest.raises(ValueError, match="Unsupported scheduler type"):
        adapter._build_scheduler({})
