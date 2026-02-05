"""Tick-based scheduler for real-time arena games."""

from __future__ import annotations

import time
from typing import Optional, Sequence

from gage_eval.role.arena.interfaces import ArenaEnvironment, Player, Scheduler
from gage_eval.role.arena.types import ArenaAction
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
        """Run a fixed-timestep loop and return the final result.

        This scheduler is intended for environments where `apply()` advances the
        simulation by a single tick. Players may compute decisions asynchronously
        and provide actions via `has_action()` / `pop_action()`.
        """

        if not players:
            raise ValueError("TickScheduler requires at least one player")

        environment.reset()

        players_by_name = {player.name: player for player in players}
        if not players_by_name:
            raise ValueError("TickScheduler requires at least one named player")

        def _noop_action(player_id: str) -> ArenaAction:
            return ArenaAction(player=str(player_id), move="noop", raw="noop", metadata={"source": "tick_scheduler"})

        tick_s = self._tick_ms / 1000.0
        ticks = 0
        next_tick = time.monotonic()
        active_player_id = environment.get_active_player()
        active_player = players_by_name.get(active_player_id)
        if active_player is None:
            raise KeyError(f"Missing player '{active_player_id}' for environment")

        current_action: Optional[ArenaAction] = None
        remaining_ticks = 0

        # STEP 1: Prime async thinking when supported.
        start = getattr(active_player, "start_thinking", None)
        if callable(start):
            start(environment.observe(active_player_id), deadline_ms=self._tick_ms)

        # STEP 2: Tick loop that applies exactly one environment tick per iteration.
        while not environment.is_terminal():
            if self._max_ticks is not None and ticks >= self._max_ticks:
                return environment.build_result(result="draw", reason="max_ticks")

            # STEP 2.1: Refresh the active player and ensure their next decision is in-flight.
            active_player_id = environment.get_active_player()
            active_player = players_by_name.get(active_player_id)
            if active_player is None:
                raise KeyError(f"Missing player '{active_player_id}' for environment")

            observation = environment.observe(active_player_id)
            if remaining_ticks <= 0:
                has_action = getattr(active_player, "has_action", None)
                pop_action = getattr(active_player, "pop_action", None)
                if callable(has_action) and callable(pop_action) and has_action():
                    decision_action = pop_action()
                    current_action = decision_action
                    hold_ticks = getattr(decision_action, "hold_ticks", None)
                    try:
                        remaining_ticks = max(1, int(hold_ticks) if hold_ticks is not None else 1)
                    except (TypeError, ValueError):
                        remaining_ticks = 1
                    record_decision = getattr(environment, "record_decision", None)
                    if callable(record_decision):
                        record_decision(
                            decision_action,
                            start_tick=ticks,
                            hold_ticks=remaining_ticks,
                        )
                else:
                    # Ensure we kick off background thinking for the next decision.
                    start = getattr(active_player, "start_thinking", None)
                    if callable(start):
                        start(observation, deadline_ms=self._tick_ms)

            # STEP 2.2: Apply the currently held action (or noop while waiting).
            action_to_apply = current_action if current_action is not None else _noop_action(active_player_id)
            outcome = environment.apply(action_to_apply)
            if remaining_ticks > 0:
                remaining_ticks -= 1
            if outcome is not None:
                return outcome

            ticks += 1

            # STEP 2.3: Sleep until the next tick boundary.
            next_tick += tick_s
            sleep_for = max(0.0, next_tick - time.monotonic())
            time.sleep(sleep_for)

        return environment.build_result(result="draw", reason="terminated")
