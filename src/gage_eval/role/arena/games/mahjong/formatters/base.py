"""Formatter interfaces for Mahjong."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

class MahjongFormatter(ABC):
    """Defines the formatting interface for Mahjong observations."""

    @abstractmethod
    def format_observation(
        self,
        raw_observation: dict[str, Any],
        legal_action_ids: Sequence[int],
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        """Format a raw observation into public/private state and legal moves.

        Args:
            raw_observation: Raw observation payload from the core.
            legal_action_ids: Legal action ids for the player.

        Returns:
            Tuple of public state, private state, and legal move strings.
        """

    @abstractmethod
    def format_action(self, action_id: int) -> str:
        """Format an action id into human-readable text."""
