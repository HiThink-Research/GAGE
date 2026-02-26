"""Action encoding utilities for ViZDoom-style discrete actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class VizDoomActionCodec:
    """Encode human/LLM moves into discrete action IDs.

    Args:
        action_labels: Optional ordered labels for each action id.
        action_mapping: Optional explicit label-to-id mapping.
        default_action: Fallback action id when encoding fails.
    """

    action_labels: Optional[Iterable[str]] = None
    action_mapping: Optional[Dict[str, int]] = None
    default_action: int = 0

    def __post_init__(self) -> None:
        if self.action_labels is not None and not isinstance(self.action_labels, tuple):
            object.__setattr__(self, "action_labels", tuple(str(label) for label in self.action_labels))

    def encode(self, move: Any) -> int:
        """Encode a move into a discrete action id.

        Args:
            move: Raw move value (string/int/label).

        Returns:
            Discrete action id.
        """

        if move is None:
            return self.default_action
        if isinstance(move, int):
            return move
        if isinstance(move, str):
            stripped = move.strip()
            if stripped.isdigit():
                return int(stripped)
            if self.action_mapping and stripped in self.action_mapping:
                return int(self.action_mapping[stripped])
            if self.action_labels:
                for idx, label in enumerate(self.action_labels):
                    if stripped == str(label):
                        return idx
        try:
            return int(move)  # type: ignore[arg-type]
        except Exception:
            return self.default_action

    def legal_moves(self) -> list[int]:
        """Return the legal action ids."""

        if self.action_mapping:
            return sorted(set(self.action_mapping.values()))
        if self.action_labels:
            return list(range(len(self.action_labels)))
        return [0, 1, 2, 3, 4]
