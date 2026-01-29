"""Action encoding utilities for stable-retro environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class EncodedAction:
    """Represents an encoded retro action."""

    buttons: list[int]
    pressed: list[str]


class RetroActionCodec:
    """Encodes macro moves into stable-retro button arrays."""

    def __init__(
        self,
        buttons: Sequence[str],
        *,
        macro_map: Mapping[str, Iterable[str]] | None = None,
        legal_moves: Sequence[str] | None = None,
    ) -> None:
        """Initialize the codec.

        Args:
            buttons: Ordered list of button names from the retro environment.
            macro_map: Optional mapping from macro move name to button names.
            legal_moves: Optional list of moves to expose (defaults to macro_map keys).
        """

        self._buttons = [str(button).upper() for button in buttons]
        if not self._buttons:
            raise ValueError("retro buttons list is empty")
        default_map = self._build_default_macro_map(self._buttons)
        resolved_map = {**default_map}
        if macro_map:
            for move, combo in macro_map.items():
                resolved_map[str(move)] = [str(btn).upper() for btn in combo]
        self._macro_map = {
            move: combo
            for move, combo in resolved_map.items()
            if all(btn in self._buttons for btn in combo)
        }
        if legal_moves is None:
            self._legal_moves = sorted(self._macro_map.keys())
        else:
            filtered = [str(move) for move in legal_moves if str(move) in self._macro_map]
            self._legal_moves = filtered
        if "noop" not in self._macro_map:
            self._macro_map["noop"] = []
            if "noop" not in self._legal_moves:
                self._legal_moves.insert(0, "noop")

    def legal_moves(self) -> list[str]:
        """Return the list of legal macro moves."""

        return list(self._legal_moves)

    def encode(self, move: str) -> EncodedAction:
        """Encode the macro move into a retro action array.

        Args:
            move: Macro move string.

        Returns:
            EncodedAction payload.
        """

        move_key = str(move)
        if move_key not in self._macro_map:
            raise ValueError(f"unknown retro move '{move_key}'")
        pressed = [str(btn).upper() for btn in self._macro_map.get(move_key, [])]
        buttons = [1 if button in pressed else 0 for button in self._buttons]
        return EncodedAction(buttons=buttons, pressed=pressed)

    @staticmethod
    def _build_default_macro_map(buttons: Sequence[str]) -> dict[str, list[str]]:
        button_set = {str(button).upper() for button in buttons}
        def _available(combo: Sequence[str]) -> bool:
            return all(btn in button_set for btn in combo)

        candidates: dict[str, list[str]] = {
            "noop": [],
            "left": ["LEFT"],
            "right": ["RIGHT"],
            "up": ["UP"],
            "down": ["DOWN"],
            "jump": ["A"],
            "run": ["B"],
            "left_jump": ["LEFT", "A"],
            "right_jump": ["RIGHT", "A"],
            "left_run": ["LEFT", "B"],
            "right_run": ["RIGHT", "B"],
            "left_run_jump": ["LEFT", "A", "B"],
            "right_run_jump": ["RIGHT", "A", "B"],
            "start": ["START"],
            "select": ["SELECT"],
        }
        return {move: combo for move, combo in candidates.items() if _available(combo)}
