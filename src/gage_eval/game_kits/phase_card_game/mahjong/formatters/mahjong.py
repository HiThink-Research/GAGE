"""Mahjong-specific observation formatter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Sequence

from gage_eval.game_kits.phase_card_game.mahjong.formatters.base import (
    MahjongFormatter,
)
from gage_eval.game_kits.phase_card_game.mahjong.mapping import (
    build_action_maps,
    rlcard_card_to_code,
)


class StandardMahjongFormatter(MahjongFormatter):
    """Formats RLCard Mahjong observations."""

    def __init__(
        self,
        *,
        action_map: Optional[dict[int, str]] = None,
        player_id_map: Optional[Mapping[int, str]] = None,
    ) -> None:
        self._action_map = action_map or self._build_default_map()
        self._player_id_map = {
            int(raw_player_index): str(player_id)
            for raw_player_index, player_id in dict(player_id_map or {}).items()
        }

    def format_observation(
        self,
        raw_observation: dict[str, Any],
        legal_action_ids: Sequence[int],
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        hand_cards = []
        raw_hand = raw_observation.get("current_hand", [])
        for card in raw_hand:
            hand_cards.append(self._format_card(card))

        table_cards = []
        if "table" in raw_observation:
            for card in raw_observation["table"]:
                table_cards.append(self._format_card(card))
        meld_groups = self._format_meld_groups(raw_observation.get("players_pile", {}))
        draw_tile = hand_cards[-1] if len(hand_cards) % 3 == 2 and hand_cards else None

        public_state = {
            "discards": table_cards,
            "melds": {
                player_id: [group["label"] for group in groups]
                for player_id, groups in meld_groups.items()
            },
            "meld_groups": meld_groups,
        }

        private_state = {
            "hand": list(hand_cards),
            "hand_raw": [str(card) for card in raw_hand],
            "draw_tile": draw_tile,
        }

        legal_moves = [self.format_action(action_id) for action_id in legal_action_ids]
        return public_state, private_state, legal_moves

    def format_action(self, action_id: int) -> str:
        return self._action_map.get(int(action_id), str(action_id))

    def _format_card(self, card_obj: Any) -> str:
        return rlcard_card_to_code(card_obj)

    def _format_meld_groups(self, raw_players_pile: Any) -> dict[str, list[dict[str, Any]]]:
        if not isinstance(raw_players_pile, Mapping):
            return {}

        formatted: dict[str, list[dict[str, Any]]] = {}
        for raw_player_index, raw_groups in raw_players_pile.items():
            try:
                player_index = int(raw_player_index)
            except Exception:
                continue
            player_id = self._player_id_map.get(player_index, str(player_index))
            groups: list[dict[str, Any]] = []
            if isinstance(raw_groups, Sequence) and not isinstance(raw_groups, (str, bytes)):
                for raw_group in raw_groups:
                    tiles = self._format_meld_tiles(raw_group)
                    if not tiles:
                        continue
                    group_type = self._infer_meld_type(tiles)
                    groups.append(
                        {
                            "type": group_type,
                            "label": (
                                f"{group_type.capitalize()} {'-'.join(tiles) if group_type == 'chow' else tiles[0]}"
                            ),
                            "tiles": tiles,
                        }
                    )
            if groups:
                formatted[player_id] = groups
        return formatted

    def _format_meld_tiles(self, raw_group: Any) -> list[str]:
        if hasattr(raw_group, "cards"):
            return [self._format_card(card) for card in list(getattr(raw_group, "cards", []))]
        if isinstance(raw_group, Sequence) and not isinstance(raw_group, (str, bytes)):
            return [self._format_card(card) for card in raw_group]
        if raw_group is None:
            return []
        return [self._format_card(raw_group)]

    def _infer_meld_type(self, tiles: Sequence[str]) -> str:
        if len(tiles) == 4:
            return "kong"
        if len(tiles) == 3 and len(set(tiles)) == 1:
            return "pong"
        return "chow"

    def _build_default_map(self) -> dict[int, str]:
        action_id_to_text, _, _ = build_action_maps()
        return action_id_to_text
