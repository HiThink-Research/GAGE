"""Data contracts for Mahjong card game arena components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, TypedDict

class MahjongChatMessage(TypedDict):
    """Represents a chat line from a player."""
    player_id: str
    text: str

@dataclass
class MahjongObservation:
    """Observation payload delivered to a Mahjong player."""
    public_state: dict[str, Any]
    private_state: dict[str, Any]
    legal_moves: Sequence[str]
    player_id: str
    active_player: str
    chat_log: Sequence[MahjongChatMessage]
    ai_persona: dict[str, Any]
    last_move: Optional[str] = None
    board_text: str = ""
    
    @property
    def metadata(self) -> dict[str, Any]:
        """Compatibility shim for LLMPlayer."""
        return {
            "public_state": self.public_state,
            "private_state": self.private_state,
            "player_id": self.player_id,
            "active_player_id": self.active_player,
            "active_player": self.active_player,
            "chat_log": self.chat_log,
            "ai_persona": self.ai_persona,
            "last_move": self.last_move
        }

class MahjongAction(TypedDict, total=False):
    """Action payload returned by a Mahjong player."""
    player_id: str
    action: str
    action_id: int
    chat: str
    raw: str

class MahjongMove(TypedDict):
    """Logged move entry for Mahjong."""
    step: int
    player_id: str
    action_id: int
    action_text: str
    action_card: Optional[str]
    chat: Optional[str]
    timestamp_ms: int

class MahjongGameResult(TypedDict):
    """Outcome summary for a completed Mahjong game."""
    winner: Optional[str]
    reason: str
    final_board_html: str
    move_log: Sequence[MahjongMove]
    chat_log: Sequence[MahjongChatMessage]
    payoffs: Sequence[float]
