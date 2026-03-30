"""RLCard core implementation for Mahjong."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from gage_eval.game_kits.phase_card_game.mahjong.mapping import (
    build_action_maps,
    rlcard_action_to_display,
    rlcard_card_to_code,
)
from gage_eval.game_kits.phase_card_game.mahjong.rlcard_patches import (
    patch_rlcard_game,
)
from gage_eval.game_kits.phase_card_game.shared.rlcard_core import RLCardCore as SharedRLCardCore


class RLCardCore(SharedRLCardCore):
    """RLCard core adapter for Mahjong."""

    def __init__(self, game_type: str, *, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__(game_type, config=config)
        action_id_to_text, action_text_to_id, action_id_to_raw = build_action_maps()
        self._action_id_to_text = action_id_to_text
        self._action_text_to_id = action_text_to_id
        self._action_id_to_raw = action_id_to_raw

    def reset(self) -> None:
        """Reset the RLCard environment and apply local patches."""

        super().reset()
        patch_rlcard_game(getattr(self._env, "game", None))

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
            raw_hand = getattr(player, "hand", [])
            raw_pile = getattr(player, "pile", [])

            formatted_hand = [rlcard_card_to_code(card) for card in raw_hand]
            formatted_hand.sort()

            formatted_pile = []
            for pile_item in raw_pile:
                if hasattr(pile_item, "cards"):
                    formatted_pile.append([rlcard_card_to_code(card) for card in pile_item.cards])
                elif isinstance(pile_item, list):
                    formatted_pile.append([rlcard_card_to_code(card) for card in pile_item])
                elif hasattr(pile_item, "get_str"):
                    formatted_pile.append(rlcard_card_to_code(pile_item))
                else:
                    formatted_pile.append(str(pile_item))

            hands[idx] = {
                "hand": formatted_hand,
                "pile": formatted_pile,
            }
        return hands


__all__ = ["RLCardCore"]
