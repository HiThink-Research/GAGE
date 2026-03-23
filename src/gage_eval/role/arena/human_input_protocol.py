"""Helpers for routing human arena input safely."""

from __future__ import annotations

import json
from queue import Queue
from threading import Lock
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


class SampleActionRouter:
    """Route sample-scoped input into player-scoped queues."""

    def __init__(self, *, sample_id: str, player_ids: Sequence[str]) -> None:
        """Initialize one router for one sample."""

        normalized_player_ids = [str(player_id) for player_id in player_ids if player_id]
        if not normalized_player_ids:
            raise ValueError("SampleActionRouter requires at least one player_id")
        self._sample_id = str(sample_id)
        self._player_queues: dict[str, Queue[str]] = {
            player_id: Queue() for player_id in normalized_player_ids
        }
        self._lock = Lock()

    @property
    def player_queues(self) -> dict[str, Queue[str]]:
        """Return bound player queues."""

        with self._lock:
            return dict(self._player_queues)

    def queue_for(self, player_id: str) -> Queue[str]:
        """Return the queue for a specific player."""

        normalized_player_id = str(player_id)
        with self._lock:
            queue = self._player_queues.get(normalized_player_id)
        if queue is None:
            raise KeyError(f"Unknown human player_id: {normalized_player_id}")
        return queue

    def put(self, queued_action: Any) -> None:
        """Route one queued action to the correct player queue."""

        self._route_action(queued_action)

    def put_nowait(self, queued_action: Any) -> None:
        """Route one queued action without blocking."""

        self._route_action(queued_action)

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
            if player_id is None and len(self._player_queues) == 1:
                player_id = next(iter(self._player_queues))
            target_queue = self._player_queues.get(player_id or "")
        if target_queue is None:
            logger.debug(
                "Drop human input for sample_id={} because player_id={} is not registered",
                self._sample_id,
                player_id,
            )
            return
        normalized_payload = dict(payload)
        normalized_payload["sample_id"] = self._sample_id
        target_queue.put(dump_action_payload(normalized_payload))


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
