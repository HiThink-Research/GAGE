"""Tick-based scheduler for real-time arena games."""

from __future__ import annotations

import time
from typing import Optional, Sequence

from gage_eval.role.arena.interfaces import ArenaEnvironment, Player, Scheduler
from gage_eval.role.arena.types import GameResult


class TickScheduler(Scheduler):
    """Tick-driven scheduler for real-time or pseudo-real-time games."""

    def __init__(self, *, tick_ms: int = 100, max_ticks: Optional[int] = None) -> None:
        """Initialize the tick scheduler.

        Args:
            tick_ms: Tick interval in milliseconds.
            max_ticks: Optional maximum number of ticks before forcing a draw.
        """

        self._tick_ms = max(1, int(tick_ms))
        self._max_ticks = max_ticks if max_ticks is None else max(1, int(max_ticks))

    def run_loop(self, environment: ArenaEnvironment, players: Sequence[Player]) -> GameResult:
        """Run the tick loop until termination and return the final result."""

        if not players:
            raise ValueError("TickScheduler requires at least one player")

        environment.reset()

        # STEP 1: Initialize async thinking for players when supported.
        for player in players:
            start = getattr(player, "start_thinking", None)
            if callable(start):
                observation = environment.observe(player.name)
                start(observation, deadline_ms=self._tick_ms)

        ticks = 0

        # STEP 2: Tick loop to apply ready actions.
        while not environment.is_terminal():
            if self._max_ticks is not None and ticks >= self._max_ticks:
                return environment.build_result(result="draw", reason="max_ticks")

            time.sleep(self._tick_ms / 1000.0)

            for player in players:
                has_action = getattr(player, "has_action", None)
                pop_action = getattr(player, "pop_action", None)
                if not callable(has_action) or not callable(pop_action):
                    continue
                if has_action():
                    outcome = environment.apply(pop_action())
                    if outcome is not None:
                        return outcome

            ticks += 1

        return environment.build_result(result="draw", reason="terminated")
