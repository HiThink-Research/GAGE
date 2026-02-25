"""Retro keyboard-to-action mapper for websocket human controls."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from gage_eval.role.arena.input_mapping import BrowserKeyEvent, GameInputMapper, HumanActionEvent

_RETRO_DEFAULT_KEY_MAP = {
    "w": "UP",
    "arrowup": "UP",
    "a": "LEFT",
    "arrowleft": "LEFT",
    "s": "DOWN",
    "arrowdown": "DOWN",
    "d": "RIGHT",
    "arrowright": "RIGHT",
    "j": "A",
    "k": "B",
    "l": "SELECT",
    "enter": "START",
}

_ACTION_PRIORITY = ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT")


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
                action_name = str(raw_action).strip().upper()
                if not key_name or not action_name:
                    continue
                resolved_map[key_name] = action_name
        self._key_map = resolved_map
        self._default_hold_ticks = max(1, int(default_hold_ticks))
        self._dedup_same_move = bool(dedup_same_move)
        self._pressed_actions: dict[str, bool] = {}
        self._last_emitted_move: Optional[str] = None

    def _map_event_to_actions(
        self,
        *,
        event: BrowserKeyEvent,
        context: Mapping[str, Any],
    ) -> Sequence[HumanActionEvent]:
        # STEP 1: Update local key/action states from this browser event.
        self._update_action_state(event)

        # STEP 2: Resolve action state to a canonical retro move string.
        move = self._resolve_move()
        if not move:
            self._last_emitted_move = None
            return []
        if self._dedup_same_move and move == self._last_emitted_move:
            return []
        self._last_emitted_move = move

        # STEP 3: Emit one queue payload for the current human player.
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

    def _update_action_state(self, event: BrowserKeyEvent) -> None:
        if event.keys:
            for raw_key, pressed in event.keys.items():
                action = self._key_map.get(_normalize_key(raw_key))
                if action is None:
                    continue
                self._pressed_actions[action] = bool(pressed)
            return

        if not event.key:
            return
        action = self._key_map.get(_normalize_key(event.key))
        if action is None:
            return
        event_type = event.event_type
        if event_type in {"keydown", "key_down"}:
            self._pressed_actions[action] = True
            return
        if event_type in {"keyup", "key_up"}:
            self._pressed_actions[action] = False
            return
        pressed = event.payload.get("pressed")
        if pressed is None:
            pressed = True
        self._pressed_actions[action] = _coerce_bool(pressed, default=True)

    def _resolve_move(self) -> Optional[str]:
        active_actions = [
            action
            for action in _ACTION_PRIORITY
            if self._pressed_actions.get(action)
        ]
        if not active_actions:
            return None
        return "+".join(active_actions)


def _normalize_key(value: Any) -> str:
    key = str(value or "").strip().lower()
    return key


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

