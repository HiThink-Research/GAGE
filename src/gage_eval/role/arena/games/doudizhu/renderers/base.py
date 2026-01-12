"""Renderer interfaces for card games."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from gage_eval.role.arena.games.doudizhu.types import CardGameResult


class CardGameRenderer(ABC):
    """Defines the rendering interface for card game visualizers."""

    @abstractmethod
    def render_frame(self, frame: dict[str, Any]) -> dict[str, Any] | str:
        """Render a single frame of game state.

        Args:
            frame: Frame payload containing public/private state.

        Returns:
            Rendered frame payload.
        """

    @abstractmethod
    def save_replay(self, game_result: CardGameResult) -> dict[str, Any] | str:
        """Build a replay payload from a completed game.

        Args:
            game_result: Final game result payload.

        Returns:
            Replay payload as dict or string.
        """
