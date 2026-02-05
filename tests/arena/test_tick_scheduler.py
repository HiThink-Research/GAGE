import time

import pytest

from gage_eval.role.arena.schedulers.tick_scheduler import TickScheduler
from gage_eval.role.arena.types import ArenaAction, ArenaObservation, GameResult


class DummyTickEnv:
    def __init__(self, *, terminal_after_ticks: int) -> None:
        self._terminal_after_ticks = max(1, int(terminal_after_ticks))
        self._tick = 0
        self._terminal = False
        self.applied_moves: list[str] = []
        self.recorded: list[tuple[str, int, int]] = []

    def reset(self) -> None:
        self._tick = 0
        self._terminal = False
        self.applied_moves = []
        self.recorded = []

    def get_active_player(self) -> str:
        return "player_0"

    def observe(self, player: str) -> ArenaObservation:
        return ArenaObservation(
            board_text=f"tick={self._tick}",
            legal_moves=["noop", "right"],
            active_player=str(player),
            view={"text": f"tick={self._tick}"},
            legal_actions={"items": ["noop", "right"]},
        )

    def apply(self, action: ArenaAction) -> GameResult | None:
        self.applied_moves.append(str(action.move))
        self._tick += 1
        if self._tick >= self._terminal_after_ticks:
            self._terminal = True
            return self.build_result(result="draw", reason="terminal")
        return None

    def is_terminal(self) -> bool:
        return self._terminal

    def build_result(self, *, status: str | None = None, result: str | None = None, reason: str | None) -> GameResult:
        resolved = str(status or result or "draw")
        return GameResult(
            winner=None,
            status=resolved,
            result=resolved,
            reason=reason,
            move_count=len(self.applied_moves),
            illegal_move_count=0,
            final_board="",
            move_log=[],
        )

    def record_decision(self, action: ArenaAction, *, start_tick: int, hold_ticks: int, **_) -> None:
        self.recorded.append((action.move, int(start_tick), int(hold_ticks)))


class DummyAsyncPlayer:
    def __init__(self, name: str, actions: list[ArenaAction]) -> None:
        self.name = str(name)
        self._actions = list(actions)
        self.started = 0

    def start_thinking(self, observation: ArenaObservation, *, deadline_ms: int = 100) -> None:
        del observation
        del deadline_ms
        self.started += 1

    def has_action(self) -> bool:
        return bool(self._actions)

    def pop_action(self) -> ArenaAction:
        return self._actions.pop(0)


def test_tick_scheduler_repeats_action_for_hold_ticks(monkeypatch: pytest.MonkeyPatch):
    # Avoid real sleeping in unit tests.
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    env = DummyTickEnv(terminal_after_ticks=3)
    player = DummyAsyncPlayer(
        "player_0",
        [
            ArenaAction(player="player_0", move="right", raw="right", hold_ticks=3),
        ],
    )
    scheduler = TickScheduler(tick_ms=1, max_ticks=20)

    result = scheduler.run_loop(env, [player])

    assert result.reason == "terminal"
    assert env.applied_moves == ["right", "right", "right"]
    assert env.recorded == [("right", 0, 3)]


def test_tick_scheduler_applies_noop_while_waiting(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    env = DummyTickEnv(terminal_after_ticks=2)
    player = DummyAsyncPlayer("player_0", [])
    scheduler = TickScheduler(tick_ms=1, max_ticks=20)

    result = scheduler.run_loop(env, [player])

    assert result.reason == "terminal"
    assert env.applied_moves == ["noop", "noop"]


def test_tick_scheduler_respects_max_ticks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    env = DummyTickEnv(terminal_after_ticks=10_000)
    player = DummyAsyncPlayer("player_0", [])
    scheduler = TickScheduler(tick_ms=1, max_ticks=2)

    result = scheduler.run_loop(env, [player])

    assert result.reason == "max_ticks"
    assert result.result == "draw"

