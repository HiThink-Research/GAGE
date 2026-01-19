
"""RLCard core implementation for Mahjong.

Overrides standard RLCardCore because MahjongEnv lacks standard decode_action.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from gage_eval.role.arena.games.common.rlcard_core import RLCardCore
from gage_eval.role.arena.games.mahjong.mapping import (
    build_action_maps,
    rlcard_action_to_display,
    rlcard_card_to_code,
)


class RLCardCore(RLCardCore):
    """RLCard core adapter for Mahjong."""

    def __init__(self, game_type: str, *, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__(game_type, config=config)
        action_id_to_text, action_text_to_id, action_id_to_raw = build_action_maps()
        self._action_id_to_text = action_id_to_text
        self._action_text_to_id = action_text_to_id
        self._action_id_to_raw = action_id_to_raw

    def decode_action(self, action_id: int) -> str:
        """Decode a Mahjong action id to string."""
        resolved = self._action_id_to_text.get(int(action_id))
        if resolved:
            return resolved
        try:
            raw = self._env._decode_action(int(action_id))
            return rlcard_action_to_display(str(raw))
        except Exception:
            return f"Action_{action_id}"

    def encode_action(self, action_text: str) -> int:
        """Encode action text to id."""
        return super().encode_action(action_text)

    def get_all_hands(self) -> Optional[dict[int, Sequence[str]]]:
        """Override to return hands formatted as standard codes."""
        if not hasattr(self._env, "game") or not hasattr(self._env.game, "players"):
            return None

        hands = {}
        for idx, player in enumerate(self._env.game.players):
            # Mahjong player has .hand which is list of Card objects
            # Card object str() is like 'B1', 'C9'
            raw_hand = getattr(player, "hand", [])
            raw_pile = getattr(player, "pile", [])

            formatted_hand = [rlcard_card_to_code(card) for card in raw_hand]
            formatted_hand.sort()

            formatted_pile = []
            for p_item in raw_pile:
                # Pile items might be Meld objects or Cards or lists
                # Rlcard Mahjong often stores Melo/Pong objects or raw cards
                # We convert to string representation
                if hasattr(p_item, "cards"):  # If Meld object
                    formatted_pile.append([rlcard_card_to_code(c) for c in p_item.cards])
                elif isinstance(p_item, list):
                    formatted_pile.append([rlcard_card_to_code(c) for c in p_item])
                elif hasattr(p_item, "get_str"):
                    formatted_pile.append(rlcard_card_to_code(p_item))
                else:
                    formatted_pile.append(str(p_item))

            hands[idx] = {
                "hand": formatted_hand,
                "pile": formatted_pile,
            }
        return hands

__all__ = ["RLCardCore"]
