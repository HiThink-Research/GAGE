"""Record-style scheduler with per-step trace generation."""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

from gage_eval.role.arena.interfaces import ArenaEnvironment, Player, Scheduler
from gage_eval.role.arena.schedulers._scheduler_utils import (
    SchedulerWaitPolicy,
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
from gage_eval.role.arena.types import ArenaAction, ArenaObservation, GameResult, attach_arena_trace


class RecordScheduler(Scheduler):
    """Run fixed-cadence record scheduling while producing unified trace entries."""

    def __init__(
        self,
        *,
        record_fps: Optional[int] = 60,
        tick_ms: Optional[int] = None,
        max_ticks: Optional[int] = None,
        max_steps: Optional[int] = None,
        action_timeout_ms: Optional[int] = None,
        timeout_fallback_move: str = "NOOP",
        timeline_id: Optional[str] = None,
        trace_step_index_start: int = 0,
        trace_timestamp_clock: str = "wall_clock",
        trace_time_clock: str = "monotonic",
        trace_finalize_timing: str = "after_env_apply",
        trace_action_format: str = "flat",
    ) -> None:
        """Initialize the scheduler.

        Args:
            record_fps: Record sampling frequency when ``tick_ms`` is not set.
            tick_ms: Optional pacing interval between ticks in milliseconds.
            max_ticks: Optional upper bound on scheduling ticks.
            max_steps: Legacy alias of ``max_ticks`` for backward compatibility.
            action_timeout_ms: Optional timeout for per-player think calls.
            timeout_fallback_move: Move submitted when think call times out.
            timeline_id: Optional timeline id attached to trace records.
            trace_step_index_start: First emitted trace step index.
            trace_timestamp_clock: Clock mode for ``timestamp`` field.
            trace_time_clock: Clock mode for ``t_obs_ready_ms``/``t_action_submitted_ms``.
            trace_finalize_timing: When ``trace_state`` changes to ``done``.
            trace_action_format: Serialization mode for action fields.
        """

        self._record_fps = None if record_fps is None else max(1, int(record_fps))
        self._tick_ms = self._resolve_tick_interval_ms(
            tick_ms=tick_ms,
            record_fps=self._record_fps,
        )
        if max_ticks is not None:
            max_tick_bound = max_ticks
            self._max_tick_reason = "max_ticks"
        else:
            max_tick_bound = max_steps
            self._max_tick_reason = "max_steps"
        self._max_ticks = None if max_tick_bound is None else max(1, int(max_tick_bound))
        self._action_timeout_ms = (
            None if action_timeout_ms is None else max(1, int(action_timeout_ms))
        )
        self._timeout_fallback_move = str(timeout_fallback_move or "NOOP")
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
        self._wait_policy = SchedulerWaitPolicy()

    def close(self) -> None:
        self._wait_policy.close()

    def run_loop(self, environment: ArenaEnvironment, players: Sequence[Player]) -> GameResult:
        """Run the record-style loop and return final result.

        Args:
            environment: Arena environment instance.
            players: Registered players in this match.

        Returns:
            Final game result with ``arena_trace`` populated.
        """

        players_by_name = {player.name: player for player in players}
        if not players_by_name:
            raise ValueError("RecordScheduler requires at least one player")

        # STEP 1: Reset environment and initialize trace state.
        environment.reset()
        arena_trace: list[dict] = []
        step_index = self._trace_step_index_start
        tick_index = 0

        # STEP 2: Run the fixed-cadence record loop and emit one trace entry per tick.
        while not environment.is_terminal():
            tick_started_at = time.monotonic()

            if self._max_ticks is not None and tick_index >= self._max_ticks:
                result = environment.build_result(result="draw", reason=self._max_tick_reason)
                return attach_arena_trace(result, arena_trace)

            active_player = environment.get_active_player()
            player = players_by_name.get(active_player)
            if player is None:
                raise KeyError(f"Missing player '{active_player}' for environment")

            observation = environment.observe(active_player)
            t_obs_ready_ms = clock_ms(self._trace_time_clock)
            trace_entry = make_trace_entry(
                step_index=step_index,
                player_id=active_player,
                timestamp_ms=clock_ms(self._trace_timestamp_clock),
                t_obs_ready_ms=t_obs_ready_ms,
                timeline_id=self._timeline_id,
            )

            action, timed_out, error_type = self._collect_action_for_tick(
                player=player,
                observation=observation,
            )
            if action is None:
                fallback_reason = "timeout" if timed_out else (error_type or "think_exception")
                action = build_fallback_action(
                    player_id=active_player,
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

            outcome = environment.apply(action)
            illegal_reason = detect_illegal_reason(outcome)
            if illegal_reason:
                trace_entry["is_action_legal"] = False
                trace_entry["illegal_reason"] = illegal_reason
            if self._trace_finalize_timing == "after_env_apply":
                arena_trace.append(finalize_trace_entry(trace_entry))

            if outcome is not None:
                return attach_arena_trace(outcome, arena_trace)

            step_index += 1
            tick_index += 1
            self._sleep_until_next_tick(tick_started_at=tick_started_at)

        # STEP 3: Emit terminal result when environment exits naturally.
        result = environment.build_result(result="draw", reason="terminated")
        return attach_arena_trace(result, arena_trace)

    def _collect_action_for_tick(
        self,
        *,
        player: Player,
        observation: ArenaObservation,
    ) -> tuple[Optional[ArenaAction], bool, Optional[str]]:
        """Collect one action without blocking the scheduler cadence."""

        if self._supports_async_action_api(player):
            return self._wait_policy.wait_async_action(
                player=player,
                observation=observation,
                timeout_ms=0,
                deadline_ms=self._action_timeout_ms,
            )

        return think_with_timeout(
            player=player,
            observation=observation,
            timeout_ms=self._resolve_sync_timeout_ms(),
            wait_policy=self._wait_policy,
        )

    @staticmethod
    def _supports_async_action_api(player: Any) -> bool:
        """Return True when the player implements async polling APIs."""

        return all(
            callable(getattr(player, method_name, None))
            for method_name in ("start_thinking", "has_action", "pop_action")
        )

    @staticmethod
    def _resolve_tick_interval_ms(*, tick_ms: Optional[int], record_fps: Optional[int]) -> int:
        """Resolve fixed sampling interval for record scheduler."""

        if tick_ms is not None and int(tick_ms) > 0:
            return int(tick_ms)
        if record_fps is not None and int(record_fps) > 0:
            return max(1, int(round(1000.0 / float(record_fps))))
        return 33

    def _resolve_sync_timeout_ms(self) -> Optional[int]:
        """Resolve timeout budget for sync players within one record tick."""

        if self._action_timeout_ms is None:
            return self._tick_ms if self._tick_ms > 0 else None
        if self._tick_ms <= 0:
            return self._action_timeout_ms
        return min(self._action_timeout_ms, self._tick_ms)

    def _sleep_until_next_tick(self, *, tick_started_at: float) -> None:
        """Sleep the remaining tick budget to preserve stable sampling cadence."""

        if self._tick_ms <= 0:
            return
        elapsed_ms = (time.monotonic() - tick_started_at) * 1000.0
        remaining_ms = float(self._tick_ms) - elapsed_ms
        if remaining_ms > 0:
            time.sleep(remaining_ms / 1000.0)
