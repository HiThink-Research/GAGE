"""Multi-timeline scheduler that orchestrates lane-based action collection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from gage_eval.role.arena.interfaces import ArenaEnvironment, Player, Scheduler
from gage_eval.role.arena.schedulers._scheduler_utils import (
    SchedulerWaitPolicy,
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


@dataclass(frozen=True)
class _LaneConfig:
    """Runtime lane configuration for multi-timeline orchestration."""

    timeline_id: str
    player_ids: tuple[str, ...]
    lane_type: str
    action_timeout_ms: Optional[int]
    timeout_fallback_move: str


class MultiTimelineScheduler(Scheduler):
    """Schedule multiple timelines and apply actions in a single global loop."""

    def __init__(
        self,
        *,
        tick_ms: int = 33,
        max_ticks: Optional[int] = None,
        default_fallback_move: str = "NOOP",
        lane_registry: Optional[dict[str, dict[str, Any]]] = None,
        timelines: Optional[Sequence[dict[str, Any]]] = None,
        trace_step_index_start: int = 0,
        trace_timestamp_clock: str = "wall_clock",
        trace_time_clock: str = "monotonic",
        trace_finalize_timing: str = "after_env_apply",
        trace_action_format: str = "flat",
    ) -> None:
        """Initialize multi-timeline scheduler.

        Args:
            tick_ms: Global scheduler tick in milliseconds.
            max_ticks: Optional maximum global ticks.
            default_fallback_move: Fallback move when lane config does not provide one.
            lane_registry: Mapping from lane_ref to lane config.
            timelines: Ordered timeline configurations.
            trace_step_index_start: First emitted trace step index.
            trace_timestamp_clock: Clock mode for ``timestamp`` field.
            trace_time_clock: Clock mode for ``t_obs_ready_ms``/``t_action_submitted_ms``.
            trace_finalize_timing: When ``trace_state`` changes to ``done``.
            trace_action_format: Serialization mode for action fields.
        """

        self._tick_ms = max(1, int(tick_ms))
        self._max_ticks = max_ticks if max_ticks is None else max(1, int(max_ticks))
        self._default_fallback_move = str(default_fallback_move or "NOOP")
        self._lane_registry = dict(lane_registry or {})
        self._timelines = list(timelines or [])
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
        """Run global timeline loop and return final result.

        Args:
            environment: Arena environment instance.
            players: Registered players in this match.

        Returns:
            Final game result with ``arena_trace`` populated.
        """

        players_by_name = {player.name: player for player in players}
        if not players_by_name:
            raise ValueError("MultiTimelineScheduler requires at least one player")
        player_order = tuple(player.name for player in players)

        lanes = self._build_lanes(players_by_name.keys())
        if not lanes:
            raise ValueError("MultiTimelineScheduler requires at least one configured timeline")
        fallback_moves = self._build_player_fallback_map(lanes)

        # STEP 1: Reset environment and initialize loop state.
        environment.reset()
        arena_trace: list[dict] = []
        tick_index = 0
        trace_step_index = self._trace_step_index_start

        # STEP 2: Per global tick, collect lane actions then apply once.
        while not environment.is_terminal():
            if self._max_ticks is not None and tick_index >= self._max_ticks:
                result = environment.build_result(result="draw", reason="max_ticks")
                return attach_arena_trace(result, arena_trace)

            action_map = {}
            step_entries: list[dict[str, Any]] = []
            for lane in lanes:
                lane_player_ids = self._resolve_lane_player_ids(
                    lane=lane,
                    environment=environment,
                )
                for player_id in lane_player_ids:
                    player = players_by_name.get(player_id)
                    if player is None:
                        raise KeyError(f"Missing player '{player_id}' for timeline '{lane.timeline_id}'")

                    observation = environment.observe(player_id)
                    t_obs_ready_ms = clock_ms(self._trace_time_clock)
                    trace_entry = make_trace_entry(
                        step_index=trace_step_index,
                        player_id=player_id,
                        timestamp_ms=clock_ms(self._trace_timestamp_clock),
                        t_obs_ready_ms=t_obs_ready_ms,
                        timeline_id=lane.timeline_id,
                    )

                    action, timed_out, error_type = think_with_timeout(
                        player=player,
                        observation=observation,
                        timeout_ms=lane.action_timeout_ms,
                        wait_policy=self._wait_policy,
                    )
                    if action is None:
                        fallback_reason = "timeout" if timed_out else (error_type or "think_exception")
                        action = build_fallback_action(
                            player_id=player_id,
                            fallback_move=lane.timeout_fallback_move,
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
                    action_map[player_id] = action
                    trace_step_index += 1
            action_map = self._complete_action_map(
                action_map=action_map,
                player_order=player_order,
                fallback_moves=fallback_moves,
            )

            outcome = apply_action_map(environment, action_map)
            illegal_reason = detect_illegal_reason(outcome)
            if illegal_reason:
                for entry in step_entries:
                    entry["is_action_legal"] = False
                    entry["illegal_reason"] = illegal_reason
            if self._trace_finalize_timing == "after_env_apply":
                for entry in step_entries:
                    arena_trace.append(finalize_trace_entry(entry))
            if outcome is not None:
                return attach_arena_trace(outcome, arena_trace)

            tick_index += 1
            time.sleep(self._tick_ms / 1000.0)

        # STEP 3: Emit terminal result when environment exits naturally.
        result = environment.build_result(result="draw", reason="terminated")
        return attach_arena_trace(result, arena_trace)

    def _resolve_lane_player_ids(
        self,
        *,
        lane: _LaneConfig,
        environment: ArenaEnvironment,
    ) -> tuple[str, ...]:
        """Resolve lane collection targets by lane type.

        Args:
            lane: Runtime lane configuration.
            environment: Arena environment instance.

        Returns:
            Ordered player ids collected in the current global tick.
        """

        # TODO(zck): Split lane collection into dedicated lane executors - Implement independent
        # record/simultaneous/tick scheduling semantics instead of a single branch-based selector.
        if lane.lane_type in {"simultaneous", "tick"}:
            return lane.player_ids

        # NOTE: Record/turn lanes model a single decision owner per tick.
        active_player = environment.get_active_player()
        if active_player in lane.player_ids:
            return (active_player,)
        return (lane.player_ids[0],)

    def _build_player_fallback_map(self, lanes: Sequence[_LaneConfig]) -> dict[str, str]:
        """Build per-player fallback moves from lane configs.

        Args:
            lanes: Ordered lane configs.

        Returns:
            Mapping from player_id to fallback move.
        """

        fallback_by_player: dict[str, str] = {}
        for lane in lanes:
            for player_id in lane.player_ids:
                fallback_by_player.setdefault(player_id, lane.timeout_fallback_move)
        return fallback_by_player

    def _complete_action_map(
        self,
        *,
        action_map: dict[str, Any],
        player_order: Sequence[str],
        fallback_moves: dict[str, str],
    ) -> dict[str, Any]:
        """Fill missing player actions to produce a complete action map.

        Args:
            action_map: Action map collected from lane collectors.
            player_order: Stable player ordering for deterministic completion.
            fallback_moves: Per-player fallback move mapping from lane config.

        Returns:
            Completed action map with every player id populated.
        """

        completed = dict(action_map)
        for player_id in player_order:
            if player_id in completed:
                continue
            fallback_move = fallback_moves.get(player_id, self._default_fallback_move)
            completed[player_id] = build_fallback_action(
                player_id=player_id,
                fallback_move=fallback_move,
                reason="missing_lane_action",
            )
        return completed

    def _build_lanes(self, available_player_ids: Sequence[str]) -> list[_LaneConfig]:
        """Resolve lane references into concrete runtime lane configs.

        Args:
            available_player_ids: Player ids available in current match.

        Returns:
            Ordered lane config list.
        """

        available = {str(player_id) for player_id in available_player_ids}
        lanes: list[_LaneConfig] = []
        for index, timeline in enumerate(self._timelines):
            if not isinstance(timeline, dict):
                raise ValueError(f"Timeline at index {index} must be a mapping")
            timeline_id = str(timeline.get("timeline_id") or f"timeline_{index}")
            player_ids = tuple(str(player_id) for player_id in (timeline.get("player_ids") or []))
            if not player_ids:
                raise ValueError(f"Timeline '{timeline_id}' must declare non-empty player_ids")
            missing_players = [player_id for player_id in player_ids if player_id not in available]
            if missing_players:
                missing = ", ".join(missing_players)
                raise KeyError(f"Timeline '{timeline_id}' references unknown players: {missing}")

            lane_cfg = {}
            lane_ref = timeline.get("lane_ref")
            if lane_ref:
                lane_cfg = self._lane_registry.get(str(lane_ref), {})
                if not lane_cfg:
                    raise KeyError(f"Timeline '{timeline_id}' references unknown lane_ref '{lane_ref}'")
            lane_cfg = dict(lane_cfg)

            lane_type = str(lane_cfg.get("type") or timeline.get("type") or "record").lower()
            if lane_type not in {"record", "simultaneous", "turn", "tick"}:
                raise ValueError(f"Unsupported lane type '{lane_type}' for timeline '{timeline_id}'")

            # TODO(zck): Promote lane-specific runtime knobs from config - Wire fields like
            # frames_per_action/record_fps/keyframe_interval into per-lane execution policies.
            timeout_value = lane_cfg.get("action_timeout_ms")
            action_timeout_ms = None if timeout_value is None else max(1, int(timeout_value))
            fallback_move = str(
                lane_cfg.get("timeout_fallback_move")
                or timeline.get("timeout_fallback_move")
                or self._default_fallback_move
            )

            lanes.append(
                _LaneConfig(
                    timeline_id=timeline_id,
                    player_ids=player_ids,
                    lane_type=lane_type,
                    action_timeout_ms=action_timeout_ms,
                    timeout_fallback_move=fallback_move,
                )
            )
        return lanes
