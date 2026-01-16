"""Base interfaces for card game cores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence


class AbstractGameCore(ABC):
    """Define the minimal interface for card game rule engines."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the game to its initial state."""

    @abstractmethod
    def step(self, action_id: int) -> None:
        """Apply an action and advance the game state.

        Args:
            action_id: Encoded action identifier.
        """

    @abstractmethod
    def get_active_player_id(self) -> int:
        """Return the active player index."""

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True when the game is over."""

    @abstractmethod
    def get_legal_actions(self, player_id: Optional[int] = None) -> Sequence[int]:
        """Return legal action ids for the given player.

        Args:
            player_id: Optional player index to query. Defaults to active player.
        """

    @abstractmethod
    def get_observation(self, player_id: int) -> dict[str, Any]:
        """Return the raw observation for a player.

        Args:
            player_id: Player index.
        """

    @abstractmethod
    def decode_action(self, action_id: int) -> str:
        """Decode an action id into text.

        Args:
            action_id: Encoded action identifier.
        """

    @abstractmethod
    def encode_action(self, action_text: str) -> int:
        """Encode an action text into an action id.

        Args:
            action_text: Action text representation.
        """

    @abstractmethod
    def get_payoffs(self) -> Sequence[float]:
        """Return payoff values for all players."""

    def step_back(self) -> bool:
        """Optionally undo the last action."""

        return False

    def get_perfect_information(self) -> dict[str, Any]:
        """Return perfect-information payload when supported."""

        return {}

    def get_all_hands(self) -> Optional[dict[str, Sequence[str]]]:
        """Return all hands when the engine exposes them."""

        return None

    def get_public_cards(self) -> Optional[Sequence[str]]:
        """Return public cards/tiles when available."""

        return None

    @property
    def num_players(self) -> int:
        """Return the number of players for this core."""

        return 0
