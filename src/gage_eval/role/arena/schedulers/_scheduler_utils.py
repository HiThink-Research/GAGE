"""Shared helper functions for arena scheduler implementations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import lru_cache
import threading
import time
from queue import Queue
from typing import Any, Optional

from loguru import logger

from gage_eval.role.arena.action_trace import (
    resolve_trace_action_applied,
    sanitize_trace_action_metadata,
)
from gage_eval.role.arena.types import ArenaAction, ArenaObservation, GameResult

_ILLEGAL_REASONS = {
    "invalid_format",
    "illegal_move",
    "unknown_player",
    "wrong_player",
    "out_of_bounds",
    "occupied",
    "illegal_action",
}
_VALID_TRACE_CLOCKS = {"wall_clock", "monotonic"}
_VALID_TRACE_FINALIZE_TIMINGS = {"after_action_submit", "after_env_apply"}
_VALID_TRACE_ACTION_FORMATS = {"flat", "envelope"}
_DEFAULT_POLL_INTERVAL_MS = 20
_DEFAULT_SYNC_EXECUTOR_WORKERS = 64


def wall_clock_ms() -> int:
    """Return current wall-clock timestamp in milliseconds."""

    return int(time.time() * 1000)


def monotonic_ms() -> int:
    """Return monotonic timestamp in milliseconds."""

    return int(time.monotonic() * 1000)


def normalize_clock_name(clock: Optional[str], *, default: str) -> str:
    """Normalize trace clock mode.

    Args:
        clock: Optional clock mode value.
        default: Default clock mode when ``clock`` is empty.

    Returns:
        Normalized clock name.

    Raises:
        ValueError: If the clock mode is unsupported.
    """

    value = str(clock or default).strip().lower()
    if value not in _VALID_TRACE_CLOCKS:
        supported = ", ".join(sorted(_VALID_TRACE_CLOCKS))
        raise ValueError(f"Unsupported trace clock '{value}'. Supported: {supported}")
    return value


def normalize_trace_finalize_timing(timing: Optional[str]) -> str:
    """Normalize trace finalization timing.

    Args:
        timing: Optional finalization timing string.

    Returns:
        Normalized timing value.

    Raises:
        ValueError: If the timing value is unsupported.
    """

    value = str(timing or "after_env_apply").strip().lower()
    if value not in _VALID_TRACE_FINALIZE_TIMINGS:
        supported = ", ".join(sorted(_VALID_TRACE_FINALIZE_TIMINGS))
        raise ValueError(f"Unsupported trace finalize timing '{value}'. Supported: {supported}")
    return value


def normalize_trace_action_format(action_format: Optional[str]) -> str:
    """Normalize trace action serialization format.

    Args:
        action_format: Optional action format value.

    Returns:
        Normalized action format.

    Raises:
        ValueError: If the action format value is unsupported.
    """

    value = str(action_format or "flat").strip().lower()
    if value not in _VALID_TRACE_ACTION_FORMATS:
        supported = ", ".join(sorted(_VALID_TRACE_ACTION_FORMATS))
        raise ValueError(f"Unsupported trace action format '{value}'. Supported: {supported}")
    return value


def clock_ms(clock: str) -> int:
    """Return current milliseconds from the specified clock.

    Args:
        clock: Clock mode (`wall_clock` or `monotonic`).

    Returns:
        Current timestamp in milliseconds.
    """

    if clock == "wall_clock":
        return wall_clock_ms()
    if clock == "monotonic":
        return monotonic_ms()
    # NOTE: Callers should normalize first; keep this defensive guard for runtime safety.
    raise ValueError(f"Unsupported trace clock '{clock}'")


def make_trace_entry(
    *,
    step_index: int,
    player_id: str,
    timestamp_ms: int,
    t_obs_ready_ms: int,
    timeline_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create an in-progress trace entry with F0-required fields.

    Args:
        step_index: Sequential scheduler step index.
        player_id: Current action owner.
        timestamp_ms: Trace event timestamp.
        t_obs_ready_ms: Observation-ready timestamp.
        timeline_id: Optional timeline identifier for multi-timeline mode.

    Returns:
        A trace dictionary initialized with required fields.
    """

    entry = {
        "step_index": int(step_index),
        "trace_state": "in_progress",
        "timestamp": int(timestamp_ms),
        "player_id": str(player_id),
        "action_raw": None,
        "action_applied": None,
        "t_obs_ready_ms": int(t_obs_ready_ms),
        "t_action_submitted_ms": int(t_obs_ready_ms),
        "timeout": False,
        "is_action_legal": True,
        "retry_count": 0,
    }
    if timeline_id is not None:
        entry["timeline_id"] = str(timeline_id)
    return entry


def finalize_trace_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Mark a trace entry as done.

    Args:
        entry: Trace dictionary created by :func:`make_trace_entry`.

    Returns:
        The same dictionary with ``trace_state`` finalized.
    """

    entry["trace_state"] = "done"
    return entry


def set_trace_action_fields(
    entry: dict[str, Any],
    action: ArenaAction,
    *,
    action_format: str,
) -> None:
    """Populate action-related fields according to serialization format.

    Args:
        entry: Trace entry object.
        action: Action to serialize into trace.
        action_format: Serialization format (`flat` or `envelope`).
    """

    metadata = sanitize_trace_action_metadata(action.metadata)
    applied_value = resolve_trace_action_applied(action.move, action.metadata)
    if action_format == "flat":
        entry["action_raw"] = action.raw
        entry["action_applied"] = applied_value
        return

    if action_format == "envelope":
        entry["action_raw"] = {
            "player_id": action.player,
            "raw": action.raw,
            "metadata": metadata,
        }
        entry["action_applied"] = {
            "player_id": action.player,
            "move": applied_value,
            "metadata": metadata,
        }
        return

    # NOTE: Callers should normalize first; keep this defensive guard for runtime safety.
    raise ValueError(f"Unsupported trace action format '{action_format}'")


def detect_illegal_reason(result: Optional[GameResult]) -> Optional[str]:
    """Infer illegal-reason tag from a terminal GameResult.

    Args:
        result: Optional game result returned by environment.apply.

    Returns:
        Illegal reason string when it matches known reasons, otherwise None.
    """

    if result is None or not result.reason:
        return None
    reason = str(result.reason)
    if reason in _ILLEGAL_REASONS:
        return reason
    return None


def infer_retry_count(action: ArenaAction) -> int:
    """Extract retry count from action metadata when present.

    Args:
        action: Parsed action object returned by player.

    Returns:
        Non-negative retry count.
    """

    value = action.metadata.get("retry_count") if isinstance(action.metadata, dict) else 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def infer_legality(action: ArenaAction) -> bool:
    """Infer legality from action payload before environment feedback.

    Args:
        action: Parsed action object returned by player.

    Returns:
        ``True`` when action appears parse-valid, otherwise ``False``.
    """

    if not str(action.move or "").strip():
        return False
    if isinstance(action.metadata, dict) and action.metadata.get("error"):
        return False
    return True


def build_fallback_action(
    *,
    player_id: str,
    fallback_move: str,
    reason: str,
) -> ArenaAction:
    """Build a scheduler fallback action.

    Args:
        player_id: Player identifier.
        fallback_move: Move text submitted on fallback.
        reason: Fallback reason tag (for example: ``timeout``).

    Returns:
        Fallback action with standardized metadata.
    """

    return ArenaAction(
        player=str(player_id),
        move=str(fallback_move),
        raw=str(fallback_move),
        metadata={
            "fallback": str(reason),
            "error": str(reason),
            "retry_count": 0,
        },
    )


class SchedulerWaitPolicy:
    """Shared wait strategy for async polling players and sync think timeouts."""

    def __init__(
        self,
        *,
        executor: Optional[ThreadPoolExecutor] = None,
        poll_interval_ms: int = _DEFAULT_POLL_INTERVAL_MS,
    ) -> None:
        self._executor = executor or ThreadPoolExecutor(
            max_workers=_DEFAULT_SYNC_EXECUTOR_WORKERS,
            thread_name_prefix="arena-think",
        )
        self._owns_executor = executor is None
        self._poll_interval_ms = max(1, int(poll_interval_ms))

    def close(self) -> None:
        if not self._owns_executor:
            return
        self._executor.shutdown(wait=False, cancel_futures=True)

    def run_sync_think(
        self,
        *,
        player: Any,
        observation: ArenaObservation,
        timeout_ms: Optional[int],
    ) -> tuple[Optional[ArenaAction], bool, Optional[str]]:
        if timeout_ms is None:
            try:
                return player.think(observation), False, None
            except Exception as exc:
                logger.warning(
                    "Scheduler think failed for player {} without timeout: {}",
                    getattr(player, "name", "<unknown>"),
                    exc,
                )
                return None, False, "think_exception"

        timeout_s = max(0.0, float(timeout_ms) / 1000.0)
        if timeout_s <= 0:
            return None, True, "timeout"

        future = self._executor.submit(player.think, observation)
        try:
            return future.result(timeout=timeout_s), False, None
        except FutureTimeoutError:
            return None, True, "timeout"
        except Exception as exc:
            logger.warning(
                "Scheduler think failed for player {} under timeout control: {}",
                getattr(player, "name", "<unknown>"),
                exc,
            )
            return None, False, "think_exception"

    def wait_async_action(
        self,
        *,
        player: Any,
        observation: ArenaObservation,
        timeout_ms: Optional[int],
        deadline_ms: Optional[int] = None,
    ) -> tuple[Optional[ArenaAction], bool, Optional[str]]:
        ready, action, error_type = self._try_pop_ready_action(player)
        if error_type is not None:
            return None, False, error_type
        if ready:
            return action, False, None

        start_thinking = getattr(player, "start_thinking")
        resolved_deadline_ms = timeout_ms if deadline_ms is None else deadline_ms
        try:
            start_thinking(observation, deadline_ms=resolved_deadline_ms)
        except Exception as exc:
            logger.warning(
                "Scheduler async think start failed for player {}: {}",
                getattr(player, "name", "<unknown>"),
                exc,
            )
            return None, False, "think_exception"

        if timeout_ms is not None and int(timeout_ms) <= 0:
            ready, action, error_type = self._try_pop_ready_action(player)
            if error_type is not None:
                return None, False, error_type
            if ready:
                return action, False, None
            return None, True, "timeout"

        wait_for_action = getattr(player, "wait_for_action", None)
        if callable(wait_for_action):
            try:
                waited = wait_for_action(timeout_ms=timeout_ms)
            except TypeError:
                waited = wait_for_action(timeout_ms)
            except Exception as exc:
                logger.warning(
                    "Scheduler native async wait failed for player {}: {}",
                    getattr(player, "name", "<unknown>"),
                    exc,
                )
                return None, False, "think_exception"
            if isinstance(waited, ArenaAction):
                return waited, False, None
            ready, action, error_type = self._try_pop_ready_action(player)
            if error_type is not None:
                return None, False, error_type
            if ready:
                return action, False, None
            if timeout_ms is not None:
                return None, True, "timeout"
            return None, False, "empty_result"

        deadline_monotonic: Optional[float] = None
        if timeout_ms is not None:
            deadline_monotonic = time.monotonic() + max(0.0, float(timeout_ms) / 1000.0)

        while True:
            ready, action, error_type = self._try_pop_ready_action(player)
            if error_type is not None:
                return None, False, error_type
            if ready:
                return action, False, None

            if deadline_monotonic is not None:
                remaining_s = deadline_monotonic - time.monotonic()
                if remaining_s <= 0:
                    return None, True, "timeout"
                time.sleep(min(self._poll_interval_ms / 1000.0, remaining_s))
                continue

            time.sleep(self._poll_interval_ms / 1000.0)

    @staticmethod
    def _try_pop_ready_action(
        player: Any,
    ) -> tuple[bool, Optional[ArenaAction], Optional[str]]:
        has_action = getattr(player, "has_action")
        pop_action = getattr(player, "pop_action")
        try:
            if not has_action():
                return False, None, None
        except Exception as exc:
            logger.warning(
                "Scheduler async action polling failed for player {}: {}",
                getattr(player, "name", "<unknown>"),
                exc,
            )
            return False, None, "think_exception"

        try:
            return True, pop_action(), None
        except Exception as exc:
            logger.warning(
                "Scheduler async action pop failed for player {}: {}",
                getattr(player, "name", "<unknown>"),
                exc,
            )
            return False, None, "think_exception"


@lru_cache(maxsize=1)
def _default_wait_policy() -> SchedulerWaitPolicy:
    return SchedulerWaitPolicy()


def think_with_timeout(
    *,
    player: Any,
    observation: ArenaObservation,
    timeout_ms: Optional[int],
    wait_policy: Optional[SchedulerWaitPolicy] = None,
) -> tuple[Optional[ArenaAction], bool, Optional[str]]:
    """Run player.think with optional timeout enforcement.

    Args:
        player: Player object implementing ``think``.
        observation: Observation payload to feed the player.
        timeout_ms: Optional timeout in milliseconds.

    Returns:
        Tuple ``(action, timed_out, error_type)``.
    """

    policy = wait_policy or _default_wait_policy()
    if _supports_async_action_api(player):
        return policy.wait_async_action(
            player=player,
            observation=observation,
            timeout_ms=timeout_ms,
        )
    return policy.run_sync_think(
        player=player,
        observation=observation,
        timeout_ms=timeout_ms,
    )


def _supports_async_action_api(player: Any) -> bool:
    return all(
        callable(getattr(player, method_name, None))
        for method_name in ("start_thinking", "has_action", "pop_action")
    )


def apply_action_map(environment: Any, action_map: dict[str, ArenaAction]) -> Optional[GameResult]:
    """Apply action map once when supported, otherwise sequentially.

    Args:
        environment: Environment instance from arena adapter.
        action_map: Mapping from player_id to ArenaAction.

    Returns:
        Terminal ``GameResult`` when produced, otherwise ``None``.
    """

    try:
        return environment.apply(action_map)  # type: ignore[arg-type]
    except (AttributeError, TypeError) as exc:
        logger.debug(
            "Scheduler batch apply unsupported by {}: {}; falling back to sequential apply.",
            type(environment).__name__,
            exc,
        )

    for action in action_map.values():
        outcome = environment.apply(action)
        if outcome is not None:
            return outcome
    return None
