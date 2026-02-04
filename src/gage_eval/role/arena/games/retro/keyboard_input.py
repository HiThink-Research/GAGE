"""Keyboard input utilities for stable-retro WebSocket sessions."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Mapping, Optional, Sequence


_DEFAULT_KEY_MAP: dict[str, str] = {
    "ArrowLeft": "left",
    "ArrowRight": "right",
    "ArrowUp": "up",
    "ArrowDown": "down",
    "w": "up",
    "a": "left",
    "s": "down",
    "d": "right",
    "Enter": "start",
    "Shift": "select",
    " ": "jump",
    "z": "jump",
    "j": "jump",
    "x": "run",
    "k": "run",
    "c": "jump",
    "l": "start",
}


def build_default_key_map() -> dict[str, str]:
    """Return the default browser key mapping for retro controls."""

    return dict(_DEFAULT_KEY_MAP)


def _normalize_key(key: str) -> str:
    if len(key) == 1:
        return key.lower()
    return key


@dataclass
class KeyState:
    """Track keyboard state and resolve macro moves for retro games."""

    key_map: Mapping[str, str]
    legal_moves: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        self._lock = Lock()
        resolved_map = {
            _normalize_key(key): str(action)
            for key, action in self.key_map.items()
        }
        self._key_map = resolved_map
        actions = set(resolved_map.values())
        self._state = {action: False for action in actions}
        self._legal_moves = {str(move) for move in self.legal_moves} if self.legal_moves else None

    def update_from_payload(self, payload: Mapping[str, object]) -> None:
        """Update key states from a JSON payload.

        Args:
            payload: Mapping of browser key strings to pressed booleans.
        """

        with self._lock:
            for raw_key, value in payload.items():
                key = _normalize_key(str(raw_key))
                action = self._key_map.get(key)
                if not action:
                    continue
                self._state[action] = bool(value)

    def set_key(self, key: str, pressed: bool) -> None:
        """Set a single key state.

        Args:
            key: Browser key string.
            pressed: Whether the key is pressed.
        """

        normalized = _normalize_key(key)
        action = self._key_map.get(normalized)
        if not action:
            return
        with self._lock:
            self._state[action] = bool(pressed)

    def snapshot(self) -> dict[str, bool]:
        """Return a snapshot of the current key state."""

        with self._lock:
            return dict(self._state)

    def resolve_move(self) -> str:
        """Resolve the current key state into a retro macro move."""

        state = self.snapshot()
        if state.get("start"):
            return "start"
        if state.get("select"):
            return "select"

        left = bool(state.get("left"))
        right = bool(state.get("right"))
        up = bool(state.get("up"))
        down = bool(state.get("down"))
        jump = bool(state.get("jump"))
        run = bool(state.get("run"))

        if left and right:
            left = False
            right = False

        if left:
            move = _resolve_directional_move("left", run=run, jump=jump)
        elif right:
            move = _resolve_directional_move("right", run=run, jump=jump)
        elif up:
            move = "up"
        elif down:
            move = "down"
        elif jump:
            move = "jump"
        elif run:
            move = "run"
        else:
            move = "noop"

        if self._legal_moves and move not in self._legal_moves:
            return "noop"
        return move


def _resolve_directional_move(direction: str, *, run: bool, jump: bool) -> str:
    if run and jump:
        return f"{direction}_run_jump"
    if run:
        return f"{direction}_run"
    if jump:
        return f"{direction}_jump"
    return direction


__all__ = ["KeyState", "build_default_key_map"]
