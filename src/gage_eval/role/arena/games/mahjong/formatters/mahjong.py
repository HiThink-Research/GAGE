"""Mahjong-specific observation formatter."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from gage_eval.role.arena.games.mahjong.formatters.base import MahjongFormatter
from gage_eval.role.arena.games.mahjong.mapping import build_action_maps, rlcard_card_to_code


class StandardMahjongFormatter(MahjongFormatter):
    """Formats RLCard Mahjong observations."""

    def __init__(self, *, action_map: Optional[dict[int, str]] = None) -> None:
        self._action_map = action_map or self._build_default_map()

    def format_observation(
        self,
        raw_observation: dict[str, Any],
        legal_action_ids: Sequence[int],
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:

        # Parse Hand
        # raw_observation['current_hand'] is a list of Card objects
        hand_cards = []
        raw_hand = raw_observation.get("current_hand", [])
        for card in raw_hand:
            hand_cards.append(self._format_card(card))
        
        # Parse Discards/Table
        # In this env, table might be empty or raw strings.
        # usually raw_observation might have 'table'
        table_cards = []
        # Support both 'table' and generic keys if structure varies
        if "table" in raw_observation:
            for card in raw_observation["table"]:
                table_cards.append(self._format_card(card))

        public_state = {
            "discards": table_cards,
            # Add piles if available
            "melds": {},
        }

        private_state = {
            "hand": sorted(hand_cards),
            "hand_raw": [str(c) for c in raw_hand],
        }

        legal_moves = [self.format_action(aid) for aid in legal_action_ids]

        return public_state, private_state, legal_moves

    def format_action(self, action_id: int) -> str:
        return self._action_map.get(int(action_id), str(action_id))

    def _format_card(self, card_obj: Any) -> str:
        return rlcard_card_to_code(card_obj)

    def _build_default_map(self) -> dict[int, str]:
        action_id_to_text, _, _ = build_action_maps()
        return action_id_to_text
