"""Helpers for routing human arena input safely."""

from __future__ import annotations

import json
from collections.abc import Mapping as MappingABC
from queue import Empty
from queue import Full
from queue import Queue
from threading import Condition, Lock
from typing import Any, Mapping, Optional, Sequence

from loguru import logger


def build_action_payload(
    *,
    action: Optional[str],
    player_id: Optional[str] = None,
    sample_id: Optional[str] = None,
    raw: Optional[str] = None,
    source: Optional[str] = None,
    run_id: Optional[str] = None,
    task_id: Optional[str] = None,
    display_id: Optional[str] = None,
    chat: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a normalized queue payload for human arena input."""

    action_text = _normalize_text(action)
    raw_text = _normalize_text(raw)
    chat_text = _normalize_text(chat)
    metadata_payload = dict(metadata) if isinstance(metadata, Mapping) else {}

    if raw_text is None:
        if chat_text is not None or metadata_payload:
            raw_payload: dict[str, Any] = {}
            if action_text is not None:
                raw_payload["action"] = action_text
            if chat_text is not None:
                raw_payload["chat"] = chat_text
            raw_payload.update(metadata_payload)
            raw_text = json.dumps(raw_payload, ensure_ascii=False)
        else:
            raw_text = action_text

    payload: dict[str, Any] = {}
    if sample_id is not None:
        payload["sample_id"] = str(sample_id)
    if player_id is not None:
        payload["player_id"] = str(player_id)
    if action_text is not None:
        payload["action"] = action_text
        payload["move"] = action_text
    if raw_text is not None:
        payload["raw"] = raw_text
    if chat_text is not None:
        payload["chat"] = chat_text
    if source is not None:
        payload["source"] = str(source)
    if run_id is not None:
        payload["run_id"] = str(run_id)
    if task_id is not None:
        payload["task_id"] = str(task_id)
    if display_id is not None:
        payload["display_id"] = str(display_id)
    if metadata_payload:
        payload["metadata"] = metadata_payload
    return payload


def parse_action_payload(raw_payload: Any) -> dict[str, Any]:
    """Parse a queued human input payload into normalized fields."""

    payload: Optional[dict[str, Any]] = None
    if isinstance(raw_payload, Mapping):
        payload = dict(raw_payload)
    elif isinstance(raw_payload, str):
        stripped = raw_payload.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                payload = dict(parsed)
        if payload is None:
            return build_action_payload(action=stripped or None, raw=stripped or None)
    else:
        text = str(raw_payload).strip()
        return build_action_payload(action=text or None, raw=text or None)

    metadata = payload.get("metadata")
    return build_action_payload(
        action=_first_text(
            payload.get("action"),
            payload.get("move"),
            payload.get("text"),
            payload.get("value"),
            payload.get("raw"),
        ),
        player_id=_first_text(
            payload.get("player_id"),
            payload.get("playerId"),
            payload.get("player"),
            payload.get("player_idx"),
            payload.get("playerIdx"),
        ),
        sample_id=_first_text(payload.get("sample_id"), payload.get("sampleId")),
        raw=_first_text(payload.get("raw")),
        source=_first_text(payload.get("source")),
        run_id=_first_text(payload.get("run_id"), payload.get("runId")),
        task_id=_first_text(payload.get("task_id"), payload.get("taskId")),
        display_id=_first_text(payload.get("display_id"), payload.get("displayId")),
        chat=_first_text(payload.get("chat"), payload.get("message"), payload.get("text")),
        metadata=metadata if isinstance(metadata, Mapping) else None,
    )


def dump_action_payload(payload: Mapping[str, Any]) -> str:
    """Serialize a normalized action payload to JSON."""

    return json.dumps(dict(payload), ensure_ascii=False)


def extract_action_text(payload: Mapping[str, Any]) -> Optional[str]:
    """Resolve the best text representation for move parsing."""

    raw_text = _first_text(payload.get("raw"))
    if _looks_like_json(raw_text):
        return raw_text
    return _first_text(
        payload.get("action"),
        payload.get("move"),
        payload.get("text"),
        payload.get("value"),
        raw_text,
    )


def action_matches_route(
    payload: Mapping[str, Any],
    *,
    sample_id: Optional[str],
    player_id: Optional[str],
) -> bool:
    """Return whether a payload matches the expected sample/player route."""

    payload_sample_id = _first_text(payload.get("sample_id"))
    payload_player_id = _first_text(payload.get("player_id"))
    if sample_id is not None and payload_sample_id not in {None, sample_id}:
        return False
    if player_id is not None and payload_player_id not in {None, player_id}:
        return False
    return True


class ContinuousStateMailbox:
    """Single-consumer mailbox that keeps only the latest unread state snapshot."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._latest: str | None = None
        self._latest_input_seq: int | float | None = None
        self._has_unread = False

    def put(self, queued_action: Any) -> None:
        normalized = _normalize_mailbox_payload(queued_action)
        next_input_seq = _extract_input_sequence(normalized)
        with self._condition:
            if (
                next_input_seq is not None
                and self._latest_input_seq is not None
                and next_input_seq < self._latest_input_seq
            ):
                return
            self._latest = normalized
            if next_input_seq is not None:
                self._latest_input_seq = next_input_seq
            self._has_unread = True
            self._condition.notify_all()

    def put_nowait(self, queued_action: Any) -> None:
        self.put(queued_action)

    def get(self, timeout: float | None = None) -> str:
        with self._condition:
            if not self._has_unread:
                notified = self._condition.wait(timeout=timeout)
                if not notified and not self._has_unread:
                    raise Empty
            return self._consume_latest_locked()

    def get_nowait(self) -> str:
        with self._condition:
            if not self._has_unread:
                raise Empty
            return self._consume_latest_locked()

    def _consume_latest_locked(self) -> str:
        payload = self._latest
        self._has_unread = False
        if payload is None:
            raise Empty
        return payload


LatestActionMailbox = ContinuousStateMailbox


class PlayerActionMailbox:
    """Track both the latest state snapshot and the ordered command queue for a player."""

    def __init__(
        self,
        *,
        max_command_queue_size: int | None = None,
        queue_overflow_policy: str | None = None,
    ) -> None:
        self.latest_mailbox = ContinuousStateMailbox()
        self.command_queue: Queue[str] = Queue(maxsize=_coerce_queue_max_size(max_command_queue_size))
        self._queue_overflow_policy = _normalize_queue_overflow_policy(queue_overflow_policy)
        self._queue_overflow_count = 0
        self._lock = Lock()

    @property
    def queue_overflow_count(self) -> int:
        """Return how many queued command payloads were dropped by overflow policy."""

        with self._lock:
            return self._queue_overflow_count

    def put(self, queued_action: Any) -> None:
        self.latest_mailbox.put(queued_action)
        self._put_command(_normalize_mailbox_payload(queued_action), block=True)

    def put_nowait(self, queued_action: Any) -> None:
        self.latest_mailbox.put(queued_action)
        self._put_command(_normalize_mailbox_payload(queued_action), block=False)

    def get(self, timeout: float | None = None) -> str:
        return self.command_queue.get(timeout=timeout)

    def get_nowait(self) -> str:
        return self.command_queue.get_nowait()

    def latest_get(self, timeout: float | None = None) -> str:
        return self.latest_mailbox.get(timeout=timeout)

    def latest_get_nowait(self) -> str:
        return self.latest_mailbox.get_nowait()

    def _put_command(self, payload: str, *, block: bool) -> None:
        try:
            if block and self.command_queue.maxsize <= 0:
                self.command_queue.put(payload)
            else:
                self.command_queue.put_nowait(payload)
            return
        except Full:
            pass

        if self._queue_overflow_policy == "drop_oldest":
            try:
                self.command_queue.get_nowait()
            except Empty:
                pass
            try:
                self.command_queue.put_nowait(payload)
            except Full:
                self._record_queue_overflow()
                return
            self._record_queue_overflow()
            return

        self._record_queue_overflow()

    def _record_queue_overflow(self) -> None:
        with self._lock:
            self._queue_overflow_count += 1


class SampleActionRouter:
    """Route sample-scoped input into player-scoped queues."""

    def __init__(
        self,
        *,
        sample_id: str,
        player_ids: Sequence[str],
        realtime_input_semantics_by_player: Mapping[str, str] | None = None,
        latest_only_player_ids: Sequence[str] = (),
        max_command_queue_size: int | None = None,
        queue_overflow_policy: str | None = None,
    ) -> None:
        """Initialize one router for one sample."""

        normalized_player_ids = [str(player_id) for player_id in player_ids if player_id]
        if not normalized_player_ids:
            raise ValueError("SampleActionRouter requires at least one player_id")
        latest_only = {str(player_id) for player_id in latest_only_player_ids if player_id}
        semantics_by_player = {
            str(player_id): _normalize_realtime_input_semantics(semantics)
            for player_id, semantics in dict(realtime_input_semantics_by_player or {}).items()
            if str(player_id).strip()
        }
        for player_id in latest_only:
            semantics_by_player[player_id] = "continuous_state"
        self._sample_id = str(sample_id)
        self._player_input_semantics = {
            player_id: semantics_by_player.get(player_id, "queued_command")
            for player_id in normalized_player_ids
        }
        self._player_channels: dict[str, PlayerActionMailbox] = {
            player_id: PlayerActionMailbox(
                max_command_queue_size=max_command_queue_size,
                queue_overflow_policy=queue_overflow_policy,
            )
            for player_id in normalized_player_ids
        }
        self._lock = Lock()

    @property
    def player_queues(self) -> dict[str, Any]:
        """Return bound player queues."""

        with self._lock:
            return {
                player_id: self._queue_for_locked(player_id)
                for player_id in self._player_channels
            }

    @property
    def player_input_semantics(self) -> dict[str, str]:
        """Return resolved realtime input semantics per player."""

        with self._lock:
            return dict(self._player_input_semantics)

    def queue_for(self, player_id: str) -> Any:
        """Return the queue for a specific player."""

        normalized_player_id = str(player_id)
        with self._lock:
            queue = self._queue_for_locked(normalized_player_id)
        if queue is None:
            raise KeyError(f"Unknown human player_id: {normalized_player_id}")
        return queue

    def latest_for(self, player_id: str) -> ContinuousStateMailbox:
        """Return the latest-state mailbox for a specific player."""

        normalized_player_id = str(player_id)
        with self._lock:
            channels = self._player_channels.get(normalized_player_id)
        if channels is None:
            raise KeyError(f"Unknown human player_id: {normalized_player_id}")
        return channels.latest_mailbox

    def put(self, queued_action: Any) -> None:
        """Route one queued action to the correct player queue."""

        self._route_action(queued_action)

    def put_nowait(self, queued_action: Any) -> None:
        """Route one queued action without blocking."""

        self._route_action(queued_action)

    def runtime_diagnostics(self) -> dict[str, int]:
        """Return runtime counters for human input routing."""

        with self._lock:
            overflow_count = sum(
                channels.queue_overflow_count for channels in self._player_channels.values()
            )
        return {"queue_overflow_count": overflow_count}

    def _route_action(self, queued_action: Any) -> None:
        payload = parse_action_payload(queued_action)
        payload_sample_id = _first_text(payload.get("sample_id"))
        if payload_sample_id not in {None, self._sample_id}:
            logger.debug(
                "Skip human input for sample_id={} because router owns sample_id={}",
                payload_sample_id,
                self._sample_id,
            )
            return
        player_id = _first_text(payload.get("player_id"))
        with self._lock:
            if player_id is None and len(self._player_channels) == 1:
                player_id = next(iter(self._player_channels))
            target_channels = self._player_channels.get(player_id or "")
        if target_channels is None:
            logger.debug(
                "Drop human input for sample_id={} because player_id={} is not registered",
                self._sample_id,
                player_id,
            )
            return
        normalized_payload = dict(payload)
        normalized_payload["sample_id"] = self._sample_id
        target_channels.put(dump_action_payload(normalized_payload))

    def _queue_for_locked(self, player_id: str) -> Any | None:
        channels = self._player_channels.get(player_id)
        if channels is None:
            return None
        if self._player_input_semantics.get(player_id) == "continuous_state":
            return channels.latest_mailbox
        return channels.command_queue


def _normalize_mailbox_payload(queued_action: Any) -> str:
    if isinstance(queued_action, str):
        return queued_action
    return dump_action_payload(parse_action_payload(queued_action))


def _extract_input_sequence(queued_action: Any) -> int | float | None:
    payload = parse_action_payload(queued_action)
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        normalized = _coerce_sequence_number(metadata.get("input_seq"))
        if normalized is not None:
            return normalized
    return _coerce_sequence_number(payload.get("input_seq"))


def _coerce_sequence_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = _normalize_text(value)
    if text is None:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return number


def _first_text(*values: Any) -> Optional[str]:
    for value in values:
        normalized = _normalize_text(value)
        if normalized is not None:
            return normalized
    return None


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _looks_like_json(value: Optional[str]) -> bool:
    if value is None:
        return False
    stripped = value.strip()
    return bool(stripped) and stripped[0] in {"{", "["} and stripped[-1] in {"}", "]"}


def _normalize_realtime_input_semantics(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"continuous_state", "queued_command"}:
            return normalized
    if isinstance(value, MappingABC):
        normalized = value.get("semantics")
        return _normalize_realtime_input_semantics(normalized)
    return "queued_command"


def _normalize_queue_overflow_policy(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"drop_newest", "drop_oldest"}:
            return normalized
    return "drop_newest"


def _coerce_queue_max_size(value: Any) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, normalized)
