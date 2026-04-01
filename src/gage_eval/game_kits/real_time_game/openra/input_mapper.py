"""OpenRA typed-action mapper for browser-driven human controls."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from gage_eval.role.arena.input_mapping import BrowserKeyEvent, GameInputMapper, HumanActionEvent

_DIRECT_ACTION_KEYS = (
    "action_id",
    "actionId",
    "action",
    "move",
    "selected_action",
)
_KEYDOWN_TYPES = {"keydown", "key_down"}


class OpenRAInputMapper(GameInputMapper):
    """Translate browser UI payloads into OpenRA typed action intents."""

    def __init__(
        self,
        *,
        key_map: Optional[Mapping[str, Any]] = None,
        enforce_legal_actions: bool = True,
    ) -> None:
        resolved_key_map: dict[str, tuple[str, dict[str, Any], Optional[str]]] = {}
        if isinstance(key_map, Mapping):
            for raw_key, raw_value in key_map.items():
                key_name = _normalize_key(raw_key)
                action_id, payload, player_id = _parse_key_binding(raw_value)
                if not key_name or not action_id:
                    continue
                resolved_key_map[key_name] = (action_id, payload, player_id)
        self._key_map = resolved_key_map
        self._enforce_legal_actions = bool(enforce_legal_actions)

    def _map_event_to_actions(
        self,
        *,
        event: BrowserKeyEvent,
        context: Mapping[str, Any],
    ) -> Sequence[HumanActionEvent]:
        legal_action_ids = _extract_legal_action_ids(context)
        action_id, payload, mapped_player_id = self._resolve_action(
            event=event,
            legal_action_ids=legal_action_ids,
        )
        if not action_id:
            return []
        resolved_action_id = self._resolve_legal_action(
            action_id=action_id,
            legal_action_ids=legal_action_ids,
        )
        if resolved_action_id is None:
            return []

        player_id = str(mapped_player_id or context.get("human_player_id") or "player_0")
        raw_payload = json.dumps(
            {
                "action_id": resolved_action_id,
                "payload": dict(payload),
            },
            ensure_ascii=False,
        )
        return [
            HumanActionEvent(
                player_id=player_id,
                move=resolved_action_id,
                raw=raw_payload,
                metadata={
                    "source": "openra_ws",
                    "event_type": event.event_type,
                },
            )
        ]

    def _resolve_action(
        self,
        *,
        event: BrowserKeyEvent,
        legal_action_ids: Sequence[str],
    ) -> tuple[Optional[str], dict[str, Any], Optional[str]]:
        payload = event.payload
        for key in _DIRECT_ACTION_KEYS:
            value = payload.get(key)
            if value is None:
                continue
            action_id = str(value).strip()
            if action_id:
                return action_id, _extract_payload_mapping(payload), None

        if event.event_type in _KEYDOWN_TYPES and event.key:
            mapped = self._key_map.get(_normalize_key(event.key))
            if mapped:
                return mapped

        if len(legal_action_ids) == 1:
            return legal_action_ids[0], {}, None
        return None, {}, None

    def _resolve_legal_action(
        self,
        *,
        action_id: str,
        legal_action_ids: Sequence[str],
    ) -> Optional[str]:
        normalized = str(action_id).strip()
        if not normalized:
            return None
        if not legal_action_ids:
            return normalized
        lookup = {
            str(candidate).strip().lower(): str(candidate)
            for candidate in legal_action_ids
            if str(candidate).strip()
        }
        resolved = lookup.get(normalized.lower())
        if resolved is not None:
            return resolved
        if self._enforce_legal_actions:
            return None
        return normalized


def _extract_payload_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    nested = payload.get("payload")
    if isinstance(nested, Mapping):
        return dict(nested)
    return {}


def _extract_legal_action_ids(context: Mapping[str, Any]) -> list[str]:
    direct = context.get("legal_moves")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        return [str(item) for item in direct if str(item).strip()]

    legal_actions = context.get("legal_actions")
    if not isinstance(legal_actions, Mapping):
        return []
    items = legal_actions.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return []

    resolved: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            for key in ("id", "action_id", "actionId", "action", "move", "label", "text"):
                value = item.get(key)
                if value is None:
                    continue
                normalized = str(value).strip()
                if normalized:
                    resolved.append(normalized)
                    break
            continue
        normalized = str(item).strip()
        if normalized:
            resolved.append(normalized)
    return resolved


def _normalize_key(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"spacebar", " "}:
        return "space"
    return normalized


def _parse_key_binding(value: Any) -> tuple[Optional[str], dict[str, Any], Optional[str]]:
    if isinstance(value, Mapping):
        action_value = (
            value.get("action_id")
            or value.get("actionId")
            or value.get("action")
            or value.get("move")
        )
        action_id = str(action_value or "").strip()
        payload = dict(value.get("payload") or {}) if isinstance(value.get("payload"), Mapping) else {}
        player_value = value.get("player_id") or value.get("playerId") or value.get("player")
        player_id = str(player_value).strip() if player_value is not None else None
        if player_id == "":
            player_id = None
        return (action_id or None), payload, player_id
    action_id = str(value or "").strip()
    return (action_id or None), {}, None


__all__ = ["OpenRAInputMapper"]
