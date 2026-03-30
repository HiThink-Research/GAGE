"""Retro keyboard-to-action mapper for websocket human controls."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional, Sequence

from gage_eval.game_kits.real_time_game.retro_platformer.keyboard_input import (
    resolve_macro_move_from_action_state,
)
from gage_eval.role.arena.input_mapping import BrowserKeyEvent, GameInputMapper, HumanActionEvent

_RETRO_DEFAULT_KEY_MAP = {
    "w": "up",
    "arrowup": "up",
    "a": "left",
    "arrowleft": "left",
    "s": "down",
    "arrowdown": "down",
    "d": "right",
    "arrowright": "right",
    "j": "jump",
    "k": "run",
    "l": "select",
    "enter": "start",
    "shift": "select",
}

_BUTTON_ALIAS_TO_ACTION = {
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "a": "jump",
    "b": "run",
    "jump": "jump",
    "run": "run",
    "start": "start",
    "select": "select",
    "enter": "start",
    "shift": "select",
}

_KEY_ALIAS_TO_ACTION = {
    "w": "up",
    "a": "left",
    "s": "down",
    "d": "right",
    "j": "jump",
    "k": "run",
    "l": "select",
}

_ACTION_TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")


class RetroInputMapper(GameInputMapper):
    """Map browser keyboard state to retro macro moves."""

    def __init__(
        self,
        *,
        key_map: Optional[Mapping[str, str]] = None,
        default_hold_ticks: int = 1,
        dedup_same_move: bool = True,
    ) -> None:
        resolved_map = dict(_RETRO_DEFAULT_KEY_MAP)
        if isinstance(key_map, Mapping):
            for raw_key, raw_action in key_map.items():
                key_name = _normalize_key(raw_key)
                action_name = _normalize_action_name(raw_action)
                if not key_name or not action_name:
                    continue
                resolved_map[key_name] = action_name
        self._key_map = resolved_map
        self._default_hold_ticks = max(1, int(default_hold_ticks))
        self._dedup_same_move = bool(dedup_same_move)
        self._pressed_actions = {
            action_name: False
            for action_name in sorted(set(self._key_map.values()))
        }
        self._last_emitted_move: Optional[str] = None

    def _map_event_to_actions(
        self,
        *,
        event: BrowserKeyEvent,
        context: Mapping[str, Any],
    ) -> Sequence[HumanActionEvent]:
        move = self._resolve_direct_move(event)
        if move is None:
            changed = self._update_action_state(event)
            if not changed:
                return []
            move = self._resolve_move()
        if not move:
            self._last_emitted_move = None
            return []
        if self._dedup_same_move and move == self._last_emitted_move and not _is_repeat_keydown(event):
            return []
        self._last_emitted_move = move

        hold_ticks = _resolve_hold_ticks(event.payload, default=self._default_hold_ticks)
        player_id = str(context.get("human_player_id") or "player_0")
        raw_payload = json.dumps(
            {"move": move, "hold_ticks": hold_ticks},
            ensure_ascii=False,
        )
        return [
            HumanActionEvent(
                player_id=player_id,
                move=move,
                raw=raw_payload,
                metadata={
                    "hold_ticks": hold_ticks,
                    "source": "retro_keyboard",
                    "event_type": event.event_type,
                },
            )
        ]

    def _update_action_state(self, event: BrowserKeyEvent) -> bool:
        changed = False
        if event.keys:
            for raw_key, pressed in event.keys.items():
                action = self._key_map.get(_normalize_key(raw_key))
                if action is None:
                    continue
                next_state = bool(pressed)
                if self._pressed_actions.get(action) != next_state:
                    self._pressed_actions[action] = next_state
                    changed = True
            return changed

        if not event.key:
            return False
        action = self._key_map.get(_normalize_key(event.key))
        if action is None:
            return False
        event_type = event.event_type
        if event_type in {"keydown", "key_down"}:
            repeated = _coerce_bool(event.payload.get("repeat"), default=False)
            changed = self._pressed_actions.get(action) is not True or repeated
            self._pressed_actions[action] = True
            return changed
        if event_type in {"keyup", "key_up"}:
            changed = self._pressed_actions.get(action) is not False
            self._pressed_actions[action] = False
            return changed
        pressed = event.payload.get("pressed")
        if pressed is None:
            pressed = True
        next_state = _coerce_bool(pressed, default=True)
        changed = self._pressed_actions.get(action) != next_state
        self._pressed_actions[action] = next_state
        return changed

    def _resolve_move(self) -> Optional[str]:
        if not self._pressed_actions:
            return None
        return resolve_macro_move_from_action_state(self._pressed_actions)

    def _resolve_direct_move(self, event: BrowserKeyEvent) -> Optional[str]:
        payload = event.payload
        direct_move = payload.get("action")
        if direct_move is None:
            direct_move = payload.get("move")
        if direct_move is None:
            return None
        return _normalize_direct_move(direct_move)


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().lower()


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


def _normalize_action_name(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return ""
    return _BUTTON_ALIAS_TO_ACTION.get(normalized, "")


def _normalize_direct_move(value: Any) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"0", "noop"}:
        return "noop"

    action_state = _parse_action_tokens(raw)
    if action_state:
        return resolve_macro_move_from_action_state(action_state)

    if re.fullmatch(r"[a-z0-9_]+", lowered):
        return lowered
    return None


def _parse_action_tokens(text: str) -> dict[str, bool]:
    action_state: dict[str, bool] = {}
    tokens = _ACTION_TOKEN_PATTERN.findall(str(text or ""))
    for raw_token in tokens:
        token = str(raw_token).strip()
        if not token:
            continue
        lowered = token.lower()

        if len(token) == 1:
            if token.isupper() and lowered in {"a", "b"}:
                action_state[_BUTTON_ALIAS_TO_ACTION[lowered]] = True
                continue
            key_alias_action = _KEY_ALIAS_TO_ACTION.get(lowered)
            if key_alias_action is not None:
                action_state[key_alias_action] = True
                continue
            button_alias_action = _BUTTON_ALIAS_TO_ACTION.get(lowered)
            if button_alias_action is not None:
                action_state[button_alias_action] = True
                continue
            return {}

        action_name = _BUTTON_ALIAS_TO_ACTION.get(lowered)
        if action_name is not None:
            action_state[action_name] = True
            continue

        if all(char in _KEY_ALIAS_TO_ACTION for char in lowered):
            for char in lowered:
                action_state[_KEY_ALIAS_TO_ACTION[char]] = True
            continue
        return {}
    return action_state


def _resolve_hold_ticks(payload: Mapping[str, Any], *, default: int) -> int:
    for key in ("hold_ticks", "holdTicks"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            continue
    return max(1, int(default))


def _is_repeat_keydown(event: BrowserKeyEvent) -> bool:
    if event.event_type not in {"keydown", "key_down"}:
        return False
    return _coerce_bool(event.payload.get("repeat"), default=False)


__all__ = ["RetroInputMapper"]
