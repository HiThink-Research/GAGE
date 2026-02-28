"""Tick-based scheduler for real-time arena games."""

from __future__ import annotations

import time
from typing import Optional, Sequence

from gage_eval.role.arena.interfaces import ArenaEnvironment, Player, Scheduler
from gage_eval.role.arena.schedulers._scheduler_utils import (
    clock_ms,
    detect_illegal_reason,
    finalize_trace_entry,
    infer_legality,
    infer_retry_count,
    make_trace_entry,
    normalize_clock_name,
    normalize_trace_action_format,
    normalize_trace_finalize_timing,
    set_trace_action_fields,
)
from gage_eval.role.arena.types import GameResult, attach_arena_trace


class TickScheduler(Scheduler):
    """Tick-driven scheduler for real-time or pseudo-real-time games.

    The scheduler also emits F0-compatible ``arena_trace`` records per applied
    action to align with downstream sample-envelope requirements.
    """

    def __init__(
        self,
        *,
        tick_ms: int = 100,
        max_ticks: Optional[int] = None,
        trace_step_index_start: int = 0,
        trace_timestamp_clock: str = "wall_clock",
        trace_time_clock: str = "monotonic",
        trace_finalize_timing: str = "after_env_apply",
        trace_action_format: str = "flat",
    ) -> None:
        """Initialize the tick scheduler.

        Args:
            tick_ms: Tick interval in milliseconds.
            max_ticks: Optional maximum number of ticks before forcing a draw.
            trace_step_index_start: First emitted trace step index.
            trace_timestamp_clock: Clock mode for ``timestamp`` field.
            trace_time_clock: Clock mode for ``t_obs_ready_ms``/``t_action_submitted_ms``.
            trace_finalize_timing: When ``trace_state`` changes to ``done``.
            trace_action_format: Serialization mode for action fields.
        """

        self._tick_ms = max(1, int(tick_ms))
        self._max_ticks = max_ticks if max_ticks is None else max(1, int(max_ticks))
        self._trace_step_index_start = int(trace_step_index_start)
        self._trace_timestamp_clock = normalize_clock_name(
            trace_timestamp_clock,
            default="wall_clock",
        )
        self._trace_time_clock = normalize_clock_name(
            trace_time_clock,
            default="monotonic",
        )
        self._trace_finalize_timing = normalize_trace_finalize_timing(trace_finalize_timing)
        self._trace_action_format = normalize_trace_action_format(trace_action_format)

    def run_loop(self, environment: ArenaEnvironment, players: Sequence[Player]) -> GameResult:
        """Run the tick loop until termination and return the final result."""

        if not players:
            raise ValueError("TickScheduler requires at least one player")

        # STEP 1: Reset environment and initialize async workers.
        environment.reset()
        arena_trace: list[dict[str, object]] = []
        step_index = self._trace_step_index_start

        for player in players:
            self._maybe_start_async_thinking(environment, player)

        ticks = 0

        # STEP 2: Tick loop to apply ready actions and emit trace entries.
        while not environment.is_terminal():
            if self._max_ticks is not None and ticks >= self._max_ticks:
                result = environment.build_result(result="draw", reason="max_ticks")
                return attach_arena_trace(result, arena_trace)

            time.sleep(self._tick_ms / 1000.0)

            for player in players:
                has_action = getattr(player, "has_action", None)
                pop_action = getattr(player, "pop_action", None)
                if not callable(has_action) or not callable(pop_action):
                    self._maybe_start_async_thinking(environment, player)
                    continue
                if has_action():
                    action = pop_action()
                    t_obs_ready_ms = clock_ms(self._trace_time_clock)
                    trace_entry = make_trace_entry(
                        step_index=step_index,
                        player_id=player.name,
                        timestamp_ms=clock_ms(self._trace_timestamp_clock),
                        t_obs_ready_ms=t_obs_ready_ms,
                    )
                    trace_entry["t_action_submitted_ms"] = clock_ms(self._trace_time_clock)
                    set_trace_action_fields(
                        trace_entry,
                        action,
                        action_format=self._trace_action_format,
                    )
                    trace_entry["retry_count"] = infer_retry_count(action)
                    trace_entry["is_action_legal"] = infer_legality(action)
                    if isinstance(action.metadata, dict) and action.metadata.get("error"):
                        trace_entry["illegal_reason"] = str(action.metadata.get("error"))

                    if self._trace_finalize_timing == "after_action_submit":
                        arena_trace.append(finalize_trace_entry(trace_entry))

                    outcome = environment.apply(action)
                    illegal_reason = detect_illegal_reason(outcome)
                    if illegal_reason:
                        trace_entry["is_action_legal"] = False
                        trace_entry["illegal_reason"] = illegal_reason
                    if self._trace_finalize_timing == "after_env_apply":
                        arena_trace.append(finalize_trace_entry(trace_entry))

                    step_index += 1
                    if outcome is not None:
                        return attach_arena_trace(outcome, arena_trace)
                    self._maybe_start_async_thinking(environment, player)
                    continue
                self._maybe_start_async_thinking(environment, player)

            ticks += 1

        # STEP 3: Emit terminal result when environment exits naturally.
        result = environment.build_result(result="draw", reason="terminated")
        return attach_arena_trace(result, arena_trace)

    def _maybe_start_async_thinking(self, environment: ArenaEnvironment, player: Player) -> None:
        """Start async thinking for an idle player if supported."""

        start = getattr(player, "start_thinking", None)
        if not callable(start):
            return
        observation = environment.observe(player.name)
        start(observation, deadline_ms=self._tick_ms)
