"""Doudizhu-specific observation formatter."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from gage_eval.role.arena.games.doudizhu.formatters.base import CardGameFormatter


class DoudizhuFormatter(CardGameFormatter):
    """Formats RLCard Doudizhu observations into human-readable state."""

    def __init__(
        self,
        *,
        player_id_map: Optional[Mapping[int, str]] = None,
        action_id_to_text: Optional[Mapping[int, str]] = None,
    ) -> None:
        """Initialize the formatter.

        Args:
            player_id_map: Optional mapping from integer ids to player ids.
            action_id_to_text: Optional mapping for action id decoding.
        """

        self._player_id_map = {int(key): str(value) for key, value in dict(player_id_map or {}).items()}
        self._action_id_to_text = dict(action_id_to_text or self._load_action_map())

    def format_observation(
        self,
        raw_observation: dict[str, Any],
        legal_action_ids: Sequence[int],
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        """Format the raw observation into public/private state.

        Args:
            raw_observation: Raw observation payload from RLCard.
            legal_action_ids: Legal action ids for the current player.

        Returns:
            Tuple of public state, private state, and legal move strings.
        """

        landlord_id = raw_observation.get("landlord")
        played_cards_raw = raw_observation.get("played_cards", [])
        num_cards_left = raw_observation.get("num_cards_left", [])
        seen_cards = raw_observation.get("seen_cards", "")
        trace = raw_observation.get("trace", [])
        self_id = raw_observation.get("self")

        public_state = {
            "landlord_id": self._resolve_player_id(landlord_id),
            "played_cards": self._format_played_cards(played_cards_raw),
            "num_cards_left": self._format_card_counts(num_cards_left),
            "seen_cards": self._format_cards(seen_cards),
            "trace": list(trace) if trace is not None else [],
        }

        current_hand = raw_observation.get("current_hand", "")
        private_state = {
            "self_id": self._resolve_player_id(self_id),
            "current_hand": self._format_cards(current_hand),
            "current_hand_text": self._format_hand_text(current_hand),
        }

        legal_moves = [self.format_action(action_id) for action_id in legal_action_ids]
        return public_state, private_state, legal_moves

    def format_action(self, action_id: int) -> str:
        """Format an action id into a Doudizhu action string.

        Args:
            action_id: Action identifier to format.

        Returns:
            Action string for the given id.
        """

        if action_id in self._action_id_to_text:
            return self._action_id_to_text[action_id]
        return str(action_id)

    def _resolve_player_id(self, player_id: Optional[int]) -> str:
        if player_id is None:
            return "unknown"
        return self._player_id_map.get(int(player_id), str(player_id))

    def _format_played_cards(self, played_cards: Sequence[str]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for idx, cards in enumerate(list(played_cards)):
            formatted.append(
                {
                    "player_id": self._resolve_player_id(idx),
                    "cards": self._format_cards(cards),
                }
            )
        return formatted

    def _format_card_counts(self, counts: Sequence[int]) -> dict[str, int]:
        resolved: dict[str, int] = {}
        for idx, count in enumerate(list(counts)):
            resolved[self._resolve_player_id(idx)] = int(count)
        return resolved

    def _format_hand_text(self, cards: str) -> str:
        card_list = self._format_cards(cards)
        return ", ".join(card_list)

    def _format_cards(self, cards: str) -> list[str]:
        mapping = {
            "T": "10",
            "J": "J",
            "Q": "Q",
            "K": "K",
            "A": "A",
            "2": "2",
            "B": "BlackJoker",
            "R": "RedJoker",
        }
        formatted: list[str] = []
        for char in str(cards):
            if char.isdigit():
                formatted.append(char)
            else:
                formatted.append(mapping.get(char, char))
        return formatted

    def _load_action_map(self) -> dict[int, str]:
        try:
            from rlcard.games.doudizhu import utils as doudizhu_utils
        except Exception as exc:
            raise RuntimeError("rlcard is required to decode doudizhu actions") from exc

        return {int(idx): str(action) for idx, action in enumerate(doudizhu_utils.ID_2_ACTION)}
