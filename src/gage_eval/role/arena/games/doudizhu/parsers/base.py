"""Parser interfaces for card games."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ParsedAction:
    """Represents a parsed card game action."""

    action_id: int
    action_text: str
    chat_text: Optional[str] = None
    raw: Optional[str] = None


class CardMoveParser(ABC):
    """Defines the parsing interface for card game actions."""

    @abstractmethod
    def parse(
        self,
        payload: str | Mapping[str, Any],
        *,
        legal_action_ids: Optional[Sequence[int]] = None,
    ) -> ParsedAction:
        """Parse an action payload into a normalized action id.

        Args:
            payload: Raw text or JSON payload.
            legal_action_ids: Optional legal action ids to validate against.

        Returns:
            Parsed action containing the action id and text.
        """
