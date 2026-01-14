"""Data contracts for card game arena components."""

from __future__ import annotations

from typing import Any, Optional, Sequence, TypedDict


class CardChatMessage(TypedDict):
    """Represents a chat line from a player."""

    player_id: str
    text: str


class CardGameObservation(TypedDict):
    """Observation payload delivered to a card game player."""

    public_state: dict[str, Any]
    private_state: dict[str, Any]
    legal_moves: Sequence[str]
    player_id: str
    active_player_id: str
    chat_log: Sequence[CardChatMessage]
    ai_persona: dict[str, Any]


class CardGameAction(TypedDict, total=False):
    """Action payload returned by a card game player."""

    player_id: str
    action: str
    action_id: int
    chat: str
    raw: str


class CardGameMove(TypedDict):
    """Logged move entry for card games."""

    player_id: str
    action_id: int
    action_text: str
    chat: Optional[str]


class CardGameResult(TypedDict):
    """Outcome summary for a completed card game."""

    winner: Optional[str]
    reason: str
    final_board_html: str
    move_log: Sequence[CardGameMove]
    chat_log: Sequence[CardChatMessage]
    payoffs: Sequence[float]
