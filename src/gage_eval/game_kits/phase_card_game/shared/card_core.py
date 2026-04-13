"""Compatibility helpers shared by phase-card GameKit families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gage_eval.game_kits.phase_card_game.shared.rlcard_core import RLCardCore


@dataclass(frozen=True)
class CardActionParse:
    """Result of parsing a card action string."""

    action_id: Optional[int]
    action_text: str
    error: Optional[str]


__all__ = ["CardActionParse", "RLCardCore"]
