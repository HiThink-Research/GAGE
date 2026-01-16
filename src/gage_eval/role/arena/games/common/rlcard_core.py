"""RLCard-backed core implementation shared by card games."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from gage_eval.role.arena.games.common.core_base import AbstractGameCore


class RLCardCore(AbstractGameCore):
    """Wrap RLCard environments with a common core interface."""

    def __init__(
        self, game_type: str, *, config: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the RLCard core.

        Args:
            game_type: RLCard environment identifier.
            config: Optional RLCard config dictionary.
        """

        import rlcard

        self._game_type = str(game_type)
        self._config = dict(config or {})
        self._env = rlcard.make(self._game_type, config=self._config)
        self._state: Optional[dict[str, Any]] = None
        self._active_player_id: Optional[int] = None
        self._action_text_to_id: dict[str, int] = {}
        self._action_id_to_text: dict[int, str] = {}
        self._load_action_map()
        self.reset()

    @property
    def num_players(self) -> int:
        """Return the number of players for the RLCard env."""

        return int(getattr(self._env, "num_players", 0) or 0)

    def reset(self) -> None:
        """Reset the RLCard environment and cache state."""

        state, player_id = self._env.reset()
        self._state = state
        self._active_player_id = int(player_id)

    def step(self, action_id: int) -> None:
        """Apply an action id and advance the environment.

        Args:
            action_id: Encoded action identifier.
        """

        state, player_id = self._env.step(int(action_id))
        self._state = state
        self._active_player_id = int(player_id)

    def get_active_player_id(self) -> int:
        """Return the active player index."""

        if self._active_player_id is None:
            return int(self._env.get_player_id())
        return int(self._active_player_id)

    def is_terminal(self) -> bool:
        """Return True if the RLCard env is over."""

        return bool(self._env.is_over())

    def get_legal_actions(self, player_id: Optional[int] = None) -> Sequence[int]:
        """Return legal action ids for the given player.

        Args:
            player_id: Optional player index to query.
        """

        state = self._env.get_state(
            int(player_id) if player_id is not None else self.get_active_player_id()
        )
        legal = state.get("legal_actions", {})
        if isinstance(legal, dict):
            return [int(action_id) for action_id in legal.keys()]
        return [int(action_id) for action_id in legal]

    def get_observation(self, player_id: int) -> dict[str, Any]:
        """Return the raw observation for a player.

        Args:
            player_id: Player index.
        """

        state = self._env.get_state(int(player_id))
        raw_obs = state.get("raw_obs")
        if isinstance(raw_obs, dict):
            return raw_obs
        raw_state = state.get("raw_observation")
        if isinstance(raw_state, dict):
            return raw_state
        return state

    def decode_action(self, action_id: int) -> str:
        """Decode an action id into a text representation.

        Args:
            action_id: Encoded action identifier.
        """

        if action_id in self._action_id_to_text:
            return self._action_id_to_text[action_id]
        if hasattr(self._env, "game") and hasattr(self._env.game, "decode_action"):
            try:
                return str(self._env.game.decode_action(int(action_id)))
            except Exception:
                pass
        return str(action_id)

    def encode_action(self, action_text: str) -> int:
        """Encode an action string into an action id.

        Args:
            action_text: Action string representation.
        """

        normalized = str(action_text or "").strip()
        if not normalized:
            raise ValueError("Empty action text")
        if normalized in self._action_text_to_id:
            return int(self._action_text_to_id[normalized])
        lowered = normalized.lower()
        if lowered in self._action_text_to_id:
            return int(self._action_text_to_id[lowered])
        try:
            parsed = int(normalized)
        except ValueError as exc:
            raise ValueError(f"Unknown action text: {normalized}") from exc
        return int(parsed)

    def get_payoffs(self) -> Sequence[float]:
        """Return payoff values for all players."""

        payoffs = self._env.get_payoffs()
        if hasattr(payoffs, "tolist"):
            return list(payoffs.tolist())
        if isinstance(payoffs, (list, tuple)):
            return list(payoffs)
        return [float(payoffs)]

    def get_perfect_information(self) -> dict[str, Any]:
        """Return perfect information payload when available."""

        for target in (self._env, getattr(self._env, "game", None)):
            if target is None:
                continue
            getter = getattr(target, "get_perfect_information", None)
            if callable(getter):
                try:
                    payload = getter()
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    return payload
        return {}

    def get_all_hands(self) -> Optional[dict[str, Sequence[str]]]:
        """Return all player hands when exposed by the engine."""

        game = getattr(self._env, "game", None)
        players = getattr(game, "players", None)
        if not players:
            return None
        hands: dict[str, Sequence[str]] = {}
        for idx, player in enumerate(players):
            cards = (
                getattr(player, "current_hand", None)
                or getattr(player, "hand", None)
                or getattr(player, "tiles", None)
            )
            if cards is None:
                continue
            try:
                cards_list = [str(card) for card in list(cards)]
            except TypeError:
                cards_list = [str(cards)]
            hands[str(idx)] = cards_list
        return hands or None

    def get_public_cards(self) -> Optional[Sequence[str]]:
        """Return public cards or tiles when available."""

        game = getattr(self._env, "game", None)
        public_tiles = getattr(game, "public_tiles", None)
        if public_tiles:
            try:
                return [str(tile) for tile in list(public_tiles)]
            except TypeError:
                return [str(public_tiles)]
        return None

    def _load_action_map(self) -> None:
        module_name = f"rlcard.games.{self._game_type}.utils"
        try:
            module = __import__(module_name, fromlist=["ACTION_2_ID", "ID_2_ACTION"])
        except Exception:
            return
        action_to_id = getattr(module, "ACTION_2_ID", None)
        id_to_action = getattr(module, "ID_2_ACTION", None)
        if isinstance(action_to_id, dict):
            self._action_text_to_id = {
                str(key): int(value) for key, value in action_to_id.items()
            }
        if isinstance(id_to_action, dict):
            self._action_id_to_text = {
                int(key): str(value) for key, value in id_to_action.items()
            }
        elif isinstance(id_to_action, (list, tuple)):
            self._action_id_to_text = {
                int(idx): str(value) for idx, value in enumerate(id_to_action)
            }
