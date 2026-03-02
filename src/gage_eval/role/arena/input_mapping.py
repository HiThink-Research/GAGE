"""Input mapping abstractions for websocket-driven human controls."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class BrowserKeyEvent:
    """Normalized browser key event payload."""

    event_type: str
    key: Optional[str]
    keys: dict[str, bool]
    timestamp_ms: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HumanActionEvent:
    """Mapped human action payload consumed by the arena queue."""

    player_id: str
    move: str
    raw: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_queue_payload(self) -> str:
        """Return queue payload text for action_queue transport."""

        if self.raw:
            return str(self.raw)
        hold_ticks = self.metadata.get("hold_ticks")
        if hold_ticks is None:
            return str(self.move)
        return json.dumps(
            {
                "move": self.move,
                "hold_ticks": hold_ticks,
            },
            ensure_ascii=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of this action event."""

        return {
            "player_id": self.player_id,
            "move": self.move,
            "raw": self.raw,
            "metadata": dict(self.metadata),
        }


class GameInputMapper(ABC):
    """Maps browser websocket inputs into arena action queue events."""

    def handle_browser_event(
        self,
        payload: Mapping[str, Any],
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> list[HumanActionEvent]:
        """Map a browser payload into queue-ready human actions.

        Args:
            payload: Browser websocket payload.
            context: Optional display/runtime context.

        Returns:
            A list of mapped human actions. Empty means "no action emitted".
        """

        # STEP 1: Normalize context and parse the raw browser event.
        normalized_context = dict(context or {})
        event = self._parse_browser_event(payload)
        if event is None:
            return []

        # STEP 2: Delegate event-to-action conversion to game-specific mapper.
        mapped = list(self._map_event_to_actions(event=event, context=normalized_context))
        if not mapped:
            return []

        # STEP 3: Fill fallback player id and sanitize action fields.
        fallback_player_id = str(normalized_context.get("human_player_id") or "player_0")
        normalized: list[HumanActionEvent] = []
        for action in mapped:
            player_id = str(action.player_id or fallback_player_id)
            move = str(action.move or "")
            raw = str(action.raw or move)
            normalized.append(
                HumanActionEvent(
                    player_id=player_id,
                    move=move,
                    raw=raw,
                    metadata=dict(action.metadata or {}),
                )
            )
        return normalized

    @abstractmethod
    def _map_event_to_actions(
        self,
        *,
        event: BrowserKeyEvent,
        context: Mapping[str, Any],
    ) -> Sequence[HumanActionEvent]:
        """Convert a normalized browser event into one or more actions."""

    @staticmethod
    def _parse_browser_event(payload: Mapping[str, Any]) -> Optional[BrowserKeyEvent]:
        if not isinstance(payload, Mapping):
            return None
        raw_event_type = payload.get("type") or payload.get("event_type") or payload.get("event")
        event_type = str(raw_event_type).strip().lower() if raw_event_type else ""
        key = payload.get("key")
        normalized_key = str(key).strip() if isinstance(key, str) and key.strip() else None

        normalized_keys: dict[str, bool] = {}
        keys_payload = payload.get("keys")
        if isinstance(keys_payload, Mapping):
            for raw_key, raw_state in keys_payload.items():
                key_name = str(raw_key).strip()
                if not key_name:
                    continue
                normalized_keys[key_name] = _coerce_bool(raw_state, default=False)

        if not event_type:
            if normalized_key is not None:
                event_type = "key_state"
            elif normalized_keys:
                event_type = "keys_state"
            else:
                return None

        timestamp_ms = _coerce_int(
            payload.get("timestamp_ms") or payload.get("ts_ms"),
            default=int(time.time() * 1000),
        )
        return BrowserKeyEvent(
            event_type=event_type,
            key=normalized_key,
            keys=normalized_keys,
            timestamp_ms=timestamp_ms,
            payload=dict(payload),
        )


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)

