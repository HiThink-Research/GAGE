"""Simultaneous scheduler for multi-player action collection."""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

from gage_eval.role.arena.interfaces import ArenaEnvironment, Player, Scheduler
from gage_eval.role.arena.schedulers._scheduler_utils import (
    apply_action_map,
    build_fallback_action,
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
    think_with_timeout,
)
from gage_eval.role.arena.types import GameResult, attach_arena_trace


class SimultaneousScheduler(Scheduler):
    """Collect actions from all players at one decision point."""

    def __init__(
        self,
        *,
        frames_per_action: int = 1,
        max_steps: Optional[int] = None,
        action_timeout_ms: Optional[int] = 120,
        timeout_fallback_move: str = "NOOP",
        tick_ms: int = 0,
        timeline_id: Optional[str] = None,
        trace_step_index_start: int = 0,
        trace_timestamp_clock: str = "wall_clock",
        trace_time_clock: str = "monotonic",
        trace_finalize_timing: str = "after_env_apply",
        trace_action_format: str = "flat",
    ) -> None:
        """Initialize the scheduler.

        Args:
            frames_per_action: Number of environment advances per decision step.
            max_steps: Optional upper bound on decision steps.
            action_timeout_ms: Optional timeout for per-player think calls.
            timeout_fallback_move: Fallback move when timeout or think error occurs.
            tick_ms: Optional pacing interval between decision steps.
            timeline_id: Optional timeline id attached to trace records.
            trace_step_index_start: First emitted trace step index.
            trace_timestamp_clock: Clock mode for ``timestamp`` field.
            trace_time_clock: Clock mode for ``t_obs_ready_ms``/``t_action_submitted_ms``.
            trace_finalize_timing: When ``trace_state`` changes to ``done``.
            trace_action_format: Serialization mode for action fields.
        """

        self._frames_per_action = max(1, int(frames_per_action))
        self._max_steps = max_steps if max_steps is None else max(1, int(max_steps))
        self._action_timeout_ms = (
            None if action_timeout_ms is None else max(1, int(action_timeout_ms))
        )
        self._timeout_fallback_move = str(timeout_fallback_move or "NOOP")
        self._tick_ms = max(0, int(tick_ms))
        self._timeline_id = timeline_id
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
        """Run simultaneous action collection loop.

        Args:
            environment: Arena environment instance.
            players: Registered players in this match.

        Returns:
            Final game result with ``arena_trace`` populated.
        """

        if not players:
            raise ValueError("SimultaneousScheduler requires at least one player")

        # STEP 1: Reset environment and initialize trace state.
        environment.reset()
        arena_trace: list[dict] = []
        decision_step = 0
        trace_step_index = self._trace_step_index_start

        # STEP 2: For each decision point, collect one action per player.
        while not environment.is_terminal():
            if self._max_steps is not None and decision_step >= self._max_steps:
                result = environment.build_result(result="draw", reason="max_steps")
                return attach_arena_trace(result, arena_trace)

            action_map = {}
            step_entries: list[dict[str, Any]] = []
            for player in players:
                observation = environment.observe(player.name)
                t_obs_ready_ms = clock_ms(self._trace_time_clock)
                trace_entry = make_trace_entry(
                    step_index=trace_step_index,
                    player_id=player.name,
                    timestamp_ms=clock_ms(self._trace_timestamp_clock),
                    t_obs_ready_ms=t_obs_ready_ms,
                    timeline_id=self._timeline_id,
                )

                action, timed_out, error_type = think_with_timeout(
                    player=player,
                    observation=observation,
                    timeout_ms=self._action_timeout_ms,
                )
                if action is None:
                    fallback_reason = "timeout" if timed_out else (error_type or "think_exception")
                    action = build_fallback_action(
                        player_id=player.name,
                        fallback_move=self._timeout_fallback_move,
                        reason=fallback_reason,
                    )

                trace_entry["timeout"] = bool(timed_out)
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
                    step_entries.append(arena_trace[-1])
                else:
                    step_entries.append(trace_entry)
                action_map[player.name] = action
                trace_step_index += 1

            # STEP 3: Advance environment using collected action map.
            for _ in range(self._frames_per_action):
                outcome = apply_action_map(environment, action_map)
                illegal_reason = detect_illegal_reason(outcome)
                if illegal_reason:
                    for entry in step_entries:
                        entry["is_action_legal"] = False
                        entry["illegal_reason"] = illegal_reason
                if outcome is not None:
                    if self._trace_finalize_timing == "after_env_apply":
                        for entry in step_entries:
                            arena_trace.append(finalize_trace_entry(entry))
                    return attach_arena_trace(outcome, arena_trace)
                if environment.is_terminal():
                    break

            if self._trace_finalize_timing == "after_env_apply":
                for entry in step_entries:
                    arena_trace.append(finalize_trace_entry(entry))

            decision_step += 1
            if self._tick_ms > 0:
                time.sleep(self._tick_ms / 1000.0)

        # STEP 4: Emit terminal result when environment exits naturally.
        result = environment.build_result(result="draw", reason="terminated")
        return attach_arena_trace(result, arena_trace)
