"""Parser interfaces for Mahjong."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

@dataclass(frozen=True)
class MahjongParsedAction:
    """Represents a parsed Mahjong action."""
    action_id: int
    action_text: str
    chat_text: Optional[str] = None
    raw: Optional[str] = None
    error: Optional[str] = None

    @property
    def coord(self) -> str:
        """Alias for action_text to satisfy LLMPlayer interface."""
        return self.action_text

class MahjongMoveParser(ABC):
    """Defines the parsing interface for Mahjong actions."""

    @abstractmethod
    def parse(
        self,
        payload: str | Mapping[str, Any],
        *,
        legal_action_ids: Optional[Sequence[int]] = None,
    ) -> MahjongParsedAction:
        """Parse an action payload into a normalized action id."""
