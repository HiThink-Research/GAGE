from __future__ import annotations

from collections.abc import Mapping
from queue import Empty
from threading import Event, Lock
import time
from typing import Any

from gage_eval.role.arena.core.invocation import GameArenaInvocationContext
from gage_eval.role.arena.core.errors import PlayerStopRequested
from gage_eval.role.arena.core.players import BaseBoundPlayer, PlayerBindingSpec
from gage_eval.role.arena.human_input_protocol import extract_action_text, parse_action_payload
from gage_eval.role.arena.player_drivers.base import PlayerDriver
from gage_eval.role.arena.types import ArenaAction, ArenaObservation

_INTERRUPTIBLE_QUEUE_POLL_INTERVAL_S = 0.05


class LocalHumanBoundPlayer(BaseBoundPlayer):
    def __init__(
        self,
        *,
        action_queue: Any | None = None,
        timeout_ms: int | None = None,
        timeout_fallback_move: str | None = None,
        input_semantics: str | None = None,
        stateful_actions: bool = False,
        scheduler_owned_realtime: bool = False,
        action_schema: Mapping[str, Any] | None = None,
        command_stale_after_ms: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._action_queue = action_queue
        self._timeout_ms = timeout_ms
        self._timeout_fallback_move = timeout_fallback_move
        self._input_semantics = _normalize_input_semantics(
            input_semantics,
            legacy_stateful_actions=stateful_actions,
        )
        self._uses_continuous_state = self._input_semantics == "continuous_state"
        self._scheduler_owned_realtime = bool(scheduler_owned_realtime)
        self._action_schema = dict(action_schema or {}) if isinstance(action_schema, Mapping) else {}
        resolved_stale_after_ms = _coerce_optional_int(command_stale_after_ms)
        self._command_stale_after_ms = (
            resolved_stale_after_ms
            if resolved_stale_after_ms is not None and resolved_stale_after_ms >= 0
            else None
        )
        self._last_continuous_payload: dict[str, Any] = {}
        self._last_continuous_ts_ms: int | None = None
        self._async_lock = Lock()
        self._async_inflight = False
        self._async_observation: ArenaObservation | None = None
        self._async_started_monotonic: float | None = None
        self._async_deadline_ms: int | None = None
        self._async_ready_action: ArenaAction | None = None
        self._async_pending_payload: dict[str, Any] = {}
        self._stop_requested = Event()

    def next_action(self, observation: ArenaObservation) -> ArenaAction:
        if self._stop_requested.is_set():
            raise PlayerStopRequested(f"Human player '{self.player_id}' stop requested")
        if self._scheduler_owned_realtime:
            action = self.poll_scheduler_owned_action(observation)
            if action is not None:
                return action
            raise ValueError(f"Human player '{self.player_id}' did not provide an action")
        payload = self._read_action_payload(observation)
        return self._build_action_from_payload(observation, payload)

    def poll_scheduler_owned_action(self, observation: ArenaObservation) -> ArenaAction | None:
        if not self._scheduler_owned_realtime:
            return self.next_action(observation)
        payload = self._read_scheduler_owned_payload()
        if not payload:
            return None
        if self._is_scheduler_owned_realtime_idle_payload(payload):
            self._clear_continuous_payload()
            return None
        return self._build_action_from_payload(observation, payload)

    def drain_scheduler_owned_actions(
        self,
        observation: ArenaObservation,
        *,
        max_items: int = 1,
    ) -> list[ArenaAction]:
        if not self._scheduler_owned_realtime or max_items <= 0:
            return []
        if self._uses_continuous_state:
            payload = self._read_scheduler_owned_payload()
            if self._is_scheduler_owned_realtime_idle_payload(payload):
                self._clear_continuous_payload()
                return []
            if payload:
                return [self._build_action_from_payload(observation, payload)]
            replay_payload = self._resolve_continuous_replay_payload()
            if not replay_payload:
                return []
            return [
                self._build_action_from_payload(
                    observation,
                    replay_payload,
                    remember_continuous=False,
                )
            ]

        queue = self._action_queue
        if queue is None:
            return []
        get_nowait = getattr(queue, "get_nowait", None)
        if not callable(get_nowait):
            return []

        drained_actions: list[ArenaAction] = []
        while len(drained_actions) < max_items:
            try:
                queued = get_nowait()
            except Empty:
                break
            drained_actions.append(
                self._build_action_from_payload(observation, parse_action_payload(queued))
            )
        return drained_actions

    def start_thinking(
        self,
        observation: ArenaObservation,
        *,
        deadline_ms: int | None = None,
    ) -> bool:
        if self._stop_requested.is_set():
            return False
        if self._action_queue is None:
            return False
        resolved_deadline_ms = _coerce_optional_int(deadline_ms)
        with self._async_lock:
            if self._async_inflight:
                return False
            self._async_inflight = True
            self._async_observation = observation
            self._async_started_monotonic = time.monotonic()
            self._async_deadline_ms = resolved_deadline_ms
            self._async_ready_action = None
            self._async_pending_payload = {}
        return True

    def has_action(self) -> bool:
        if self._stop_requested.is_set():
            self._clear_async_state()
            return False
        with self._async_lock:
            ready_action = self._async_ready_action
            observation = self._async_observation
            started_monotonic = self._async_started_monotonic
            deadline_ms = self._async_deadline_ms
            inflight = self._async_inflight
            pending_payload = dict(self._async_pending_payload)
        if ready_action is not None:
            return True
        if not inflight or observation is None:
            return False

        payload = self._read_async_action_payload(pending_payload)
        if payload:
            pending_payload = dict(payload)
        action = self._build_async_action(
            observation=observation,
            payload=pending_payload,
            started_monotonic=started_monotonic,
            deadline_ms=deadline_ms,
        )
        if action is None:
            with self._async_lock:
                if self._async_inflight:
                    self._async_pending_payload = dict(pending_payload)
            return False
        with self._async_lock:
            if not self._async_inflight:
                return False
            self._async_pending_payload = {}
            self._async_ready_action = action
        return True

    def pop_action(self) -> ArenaAction:
        with self._async_lock:
            ready_action = self._async_ready_action
            if ready_action is None:
                raise Empty
            self._async_ready_action = None
            self._async_inflight = False
            self._async_observation = None
            self._async_started_monotonic = None
            self._async_deadline_ms = None
            self._async_pending_payload = {}
        return ready_action

    def _read_action_payload(self, observation: ArenaObservation) -> dict[str, Any]:
        queue = self._action_queue
        if queue is not None:
            if self._uses_continuous_state:
                return self._read_latest_queued_payload(queue, timeout_s=None)
            timeout_s = None
            if self._timeout_ms is not None:
                timeout_s = max(0.0, float(self._timeout_ms) / 1000.0)
            return self._read_interruptible_payload(queue, timeout_s=timeout_s)
        if self._stop_requested.is_set():
            raise PlayerStopRequested(f"Human player '{self.player_id}' stop requested")
        return parse_action_payload(input(self._format_prompt(observation)))

    def _read_scheduler_owned_payload(self) -> dict[str, Any]:
        queue = self._action_queue
        if queue is None:
            return {}
        if self._uses_continuous_state:
            payload = self._read_latest_queued_payload(queue, timeout_s=None)
            if payload:
                return payload
            return {}
        get_nowait = getattr(queue, "get_nowait", None)
        if not callable(get_nowait):
            return {}
        try:
            queued = get_nowait()
        except Empty:
            return {}
        return parse_action_payload(queued)

    def _is_scheduler_owned_realtime_idle_payload(self, payload: Mapping[str, Any] | None) -> bool:
        if not self._scheduler_owned_realtime or not self._uses_continuous_state:
            return False
        if not isinstance(payload, Mapping):
            return False
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            return False
        if not _coerce_optional_bool(metadata.get("realtime_input"), default=False):
            return False
        move = self._resolve_move(payload)
        if not move:
            return True
        idle_moves = {"noop"}
        if self._timeout_fallback_move:
            idle_moves.add(str(self._timeout_fallback_move).strip())
        return move in idle_moves

    def _read_async_action_payload(self, pending_payload: Mapping[str, Any] | None) -> dict[str, Any]:
        queue = self._action_queue
        if queue is None:
            return {}
        if self._uses_continuous_state:
            return self._read_latest_queued_payload(queue, timeout_s=None)
        if isinstance(pending_payload, Mapping) and pending_payload:
            return dict(pending_payload)
        get_nowait = getattr(queue, "get_nowait", None)
        if not callable(get_nowait):
            return {}
        try:
            queued = get_nowait()
        except Empty:
            return {}
        return parse_action_payload(queued)

    def _read_latest_queued_payload(
        self,
        queue: Any,
        *,
        timeout_s: float | None,
    ) -> dict[str, Any]:
        latest_payload: dict[str, Any] = {}
        get = getattr(queue, "get", None)
        get_nowait = getattr(queue, "get_nowait", None)
        if timeout_s is not None and callable(get):
            try:
                queued = get(timeout=timeout_s)
            except Empty:
                return {}
            latest_payload = parse_action_payload(queued)
        elif callable(get_nowait):
            try:
                queued = get_nowait()
            except Empty:
                return {}
            latest_payload = parse_action_payload(queued)
        else:
            return {}

        if not callable(get_nowait):
            return latest_payload

        while True:
            try:
                queued = get_nowait()
            except Empty:
                return latest_payload
            latest_payload = parse_action_payload(queued)

    def _build_async_action(
        self,
        *,
        observation: ArenaObservation,
        payload: Mapping[str, Any] | None,
        started_monotonic: float | None,
        deadline_ms: int | None,
    ) -> ArenaAction | None:
        resolved_payload = dict(payload) if isinstance(payload, Mapping) else {}
        if self._uses_continuous_state:
            if resolved_payload:
                self._last_continuous_payload = dict(resolved_payload)
            elif self._last_continuous_payload:
                resolved_payload = dict(self._last_continuous_payload)
        if not self._has_async_deadline_expired(
            started_monotonic=started_monotonic,
            deadline_ms=deadline_ms,
        ):
            return None
        if self._uses_continuous_state and not resolved_payload:
            if self._last_continuous_payload:
                resolved_payload = dict(self._last_continuous_payload)
            else:
                return None

        try:
            return self._build_action_from_payload(observation, resolved_payload)
        except ValueError:
            return None

    def _resolve_move(self, payload: Mapping[str, Any] | None) -> str:
        if not isinstance(payload, Mapping):
            return ""
        move = payload.get("action", payload.get("move"))
        if move is None:
            move = extract_action_text(payload)
        return str(move or "").strip()

    def _resolve_raw(self, payload: Mapping[str, Any] | None, move: str) -> str:
        if isinstance(payload, Mapping):
            raw = str(payload.get("raw") or "").strip()
            if raw:
                return raw
        return str(move or "").strip()

    def _resolve_timeout_fallback_move(self, observation: ArenaObservation) -> str | None:
        if self._timeout_fallback_move:
            return self._timeout_fallback_move
        legal_moves = list(observation.legal_actions_items)
        if legal_moves:
            return str(legal_moves[0])
        return None

    def _build_action_from_payload(
        self,
        observation: ArenaObservation,
        payload: Mapping[str, Any] | None,
        *,
        remember_continuous: bool = True,
    ) -> ArenaAction:
        resolved_payload = dict(payload) if isinstance(payload, Mapping) else {}
        if self._uses_continuous_state:
            if resolved_payload:
                if remember_continuous:
                    self._remember_continuous_payload(resolved_payload)
            elif self._last_continuous_payload:
                resolved_payload = dict(self._last_continuous_payload)
        move = self._resolve_move(resolved_payload)
        if not move:
            fallback = self._resolve_timeout_fallback_move(observation)
            move = str(fallback or "").strip()
        if not move:
            raise ValueError(f"Human player '{self.player_id}' did not provide an action")
        metadata = {
            "driver_id": self.metadata.get("driver_id"),
            "player_type": "human",
            "input_semantics": self._input_semantics,
        }
        payload_metadata = resolved_payload.get("metadata") if isinstance(resolved_payload, Mapping) else None
        if isinstance(payload_metadata, Mapping):
            metadata.update(dict(payload_metadata))
        if "hold_ticks" not in metadata:
            payload_hold_ticks = _coerce_optional_int(
                resolved_payload.get("hold_ticks", resolved_payload.get("holdTicks"))
            )
            if payload_hold_ticks is not None:
                metadata["hold_ticks"] = payload_hold_ticks
        if "hold_ticks" not in metadata:
            default_hold_ticks = _resolve_default_hold_ticks(
                observation=observation,
                action_schema=self._action_schema,
            )
            if default_hold_ticks is not None:
                metadata["hold_ticks"] = default_hold_ticks
        return ArenaAction(
            player=self.player_id,
            move=move,
            raw=self._resolve_raw(resolved_payload, move),
            metadata=metadata,
        )

    def _has_async_deadline_expired(
        self,
        *,
        started_monotonic: float | None,
        deadline_ms: int | None,
    ) -> bool:
        if deadline_ms is None or started_monotonic is None:
            return False
        elapsed_ms = (time.monotonic() - started_monotonic) * 1000.0
        return elapsed_ms >= float(deadline_ms)

    def _format_prompt(self, observation: ArenaObservation) -> str:
        legal_hint = ", ".join(str(item) for item in observation.legal_actions_items) or "none"
        return (
            f"Active player: {observation.active_player}\n"
            f"{observation.view_text}\n"
            f"Legal moves: {legal_hint}\n"
            "Enter exactly one legal move: "
        )

    def request_stop(self) -> None:
        self._stop_requested.set()
        self._clear_continuous_payload()
        self._clear_async_state()

    def reset_runtime_state(self) -> None:
        self._stop_requested.clear()
        self._clear_continuous_payload()
        self._clear_async_state()

    def _remember_continuous_payload(self, payload: Mapping[str, Any]) -> None:
        self._last_continuous_payload = dict(payload)
        self._last_continuous_ts_ms = _monotonic_ms()

    def _clear_continuous_payload(self) -> None:
        self._last_continuous_payload = {}
        self._last_continuous_ts_ms = None

    def _resolve_continuous_replay_payload(self) -> dict[str, Any]:
        if not self._last_continuous_payload:
            return {}
        if self._command_stale_after_ms is None:
            return dict(self._last_continuous_payload)
        if self._last_continuous_ts_ms is None:
            return dict(self._last_continuous_payload)
        age_ms = _monotonic_ms() - self._last_continuous_ts_ms
        if age_ms > self._command_stale_after_ms:
            self._clear_continuous_payload()
            return {}
        return dict(self._last_continuous_payload)

    def _clear_async_state(self) -> None:
        with self._async_lock:
            self._async_inflight = False
            self._async_observation = None
            self._async_started_monotonic = None
            self._async_deadline_ms = None
            self._async_ready_action = None
            self._async_pending_payload = {}

    def _read_interruptible_payload(
        self,
        queue: Any,
        *,
        timeout_s: float | None,
    ) -> dict[str, Any]:
        get = getattr(queue, "get", None)
        if not callable(get):
            return {}

        deadline_monotonic = None if timeout_s is None else time.monotonic() + max(0.0, float(timeout_s))
        while True:
            if self._stop_requested.is_set():
                raise PlayerStopRequested(f"Human player '{self.player_id}' stop requested")

            wait_timeout = _INTERRUPTIBLE_QUEUE_POLL_INTERVAL_S
            if deadline_monotonic is not None:
                remaining_s = deadline_monotonic - time.monotonic()
                if remaining_s <= 0:
                    return {}
                wait_timeout = min(wait_timeout, remaining_s)

            try:
                queued = get(timeout=wait_timeout)
            except Empty:
                if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                    return {}
                continue
            return parse_action_payload(queued)


class LocalHumanInputDriver(PlayerDriver):
    def bind(
        self,
        spec: PlayerBindingSpec,
        *,
        invocation: GameArenaInvocationContext | None = None,
    ) -> LocalHumanBoundPlayer:
        params = self.resolve_params(spec)
        action_queue = params.get("action_queue")
        if action_queue is None and invocation is not None:
            action_queue = invocation.queue_for_player(spec.player_id)
        timeout_ms = _coerce_optional_int(params.get("timeout_ms"))
        timeout_fallback_move = params.get("timeout_fallback_move")
        return LocalHumanBoundPlayer(
            player_id=spec.player_id,
            display_name=spec.player_id,
            seat=spec.seat,
            player_kind=spec.player_kind,
            action_queue=action_queue,
            timeout_ms=timeout_ms,
            timeout_fallback_move=(
                None if timeout_fallback_move is None else str(timeout_fallback_move)
            ),
            input_semantics=params.get("input_semantics") or params.get("realtime_input_semantics"),
            stateful_actions=_coerce_optional_bool(params.get("stateful_actions"), default=False),
            scheduler_owned_realtime=_coerce_optional_bool(
                params.get("scheduler_owned_realtime"),
                default=False,
            ),
            action_schema=(
                params.get("action_schema")
                if isinstance(params.get("action_schema"), Mapping)
                else None
            ),
            command_stale_after_ms=_coerce_optional_int(params.get("command_stale_after_ms")),
            metadata={"driver_id": self.driver_id, "seat": spec.seat},
        )


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _monotonic_ms() -> int:
    return int(time.monotonic() * 1000.0)


def _resolve_default_hold_ticks(
    *,
    observation: ArenaObservation,
    action_schema: Mapping[str, Any],
) -> int | None:
    candidates: list[Mapping[str, Any]] = []
    if action_schema:
        candidates.append(action_schema)
    metadata = observation.metadata if isinstance(observation.metadata, Mapping) else {}
    schema_config = metadata.get("action_schema_config")
    if isinstance(schema_config, Mapping):
        candidates.append(schema_config)
    context = observation.context if isinstance(observation.context, Mapping) else {}
    context_schema = context.get("action_schema_config")
    if isinstance(context_schema, Mapping):
        candidates.append(context_schema)
    prompt = observation.prompt
    prompt_payload = prompt.payload if prompt is not None and isinstance(prompt.payload, Mapping) else {}
    prompt_schema = prompt_payload.get("action_schema_config")
    if isinstance(prompt_schema, Mapping):
        candidates.append(prompt_schema)

    for candidate in candidates:
        default_hold_ticks = _coerce_optional_int(
            candidate.get("hold_ticks_default", candidate.get("default_hold_ticks"))
        )
        if default_hold_ticks is not None:
            return default_hold_ticks
    return None


def _normalize_input_semantics(
    value: object,
    *,
    legacy_stateful_actions: bool,
) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"continuous_state", "queued_command"}:
            return normalized
    if legacy_stateful_actions:
        return "continuous_state"
    return "queued_command"
