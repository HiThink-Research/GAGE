"""Renderer interfaces for Mahjong."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from gage_eval.role.arena.games.mahjong.types import MahjongGameResult

class MahjongRenderer(ABC):
    """Defines the rendering interface for Mahjong views."""

    @abstractmethod
    def render_frame(self, frame: dict[str, Any]) -> dict[str, Any] | str:
        """Render a single frame of game state."""

    @abstractmethod
    def save_replay(self, game_result: MahjongGameResult) -> dict[str, Any] | str:
        """Build a replay payload from a completed game."""
