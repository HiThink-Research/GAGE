"""RLCard-backed core engine for card game arenas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from gage_eval.role.arena.games.card.renderer import CardGameBoardRenderer
from gage_eval.role.arena.games.card.types import CardGameMove, CardGameObservation
from gage_eval.role.arena.games.card.formatters import CardFormatter, load_formatter


@dataclass(frozen=True)
class CardActionParse:
    """Result of parsing a card action string."""

    action_id: Optional[int]
    action_text: str
    error: Optional[str]


class RLCardCore:
    """Wrap RLCard environments with formatter and renderer helpers."""

    def __init__(
        self,
        *,
        game_type: str,
        config: Optional[dict[str, Any]] = None,
        formatter: Optional[CardFormatter] = None,
    ) -> None:
        """Initialize the RLCard environment and formatter.

        Args:
            game_type: RLCard environment id (e.g., "doudizhu").
            config: Optional RLCard config payload.
            formatter: Optional formatter instance for the game.
        """

        import rlcard

        self.game_type = str(game_type)
        self._config = dict(config or {})
        self._env = rlcard.make(self.game_type, config=self._config)
        self._formatter = formatter or load_formatter(self.game_type, env=self._env)
        self._renderer = CardGameBoardRenderer()
        self._player_ids = [f"player_{idx}" for idx in range(self._env.num_players)]
        self._player_names = {player_id: player_id for player_id in self._player_ids}
        self._state: Optional[dict[str, Any]] = None
        self._active_player_id: Optional[int] = None
        self._move_log: list[CardGameMove] = []
        self._last_action: Optional[CardGameMove] = None
        self.reset()

    @property
    def num_players(self) -> int:
        """Return the number of players for the underlying RLCard env."""

        return int(self._env.num_players)

    def configure_players(self, player_ids: Sequence[str], player_names: Optional[dict[str, str]] = None) -> None:
        """Set player identifiers and display names.

        Args:
            player_ids: Ordered player identifiers.
            player_names: Optional mapping from player_id to display name.
        """

        resolved = [str(player_id) for player_id in player_ids]
        if len(resolved) != self.num_players:
            raise ValueError(
                f"RLCard game '{self.game_type}' expects {self.num_players} players, got {len(resolved)}."
            )
        self._player_ids = resolved
        names = dict(player_names or {})
        self._player_names = {player_id: names.get(player_id, player_id) for player_id in resolved}

    def reset(self) -> None:
        """Reset the RLCard environment and clear the move log."""

        self._state, self._active_player_id = self._env.reset()
        self._move_log = []
        self._last_action = None

    def is_terminal(self) -> bool:
        """Return True when the RLCard environment reports terminal state."""

        return bool(self._env.is_over())

    def get_active_player_id(self) -> int:
        """Return the current RLCard player index."""

        return int(self._env.get_player_id())

    def step(self, action_id: int, *, chat_text: Optional[str] = None) -> CardGameMove:
        """Apply an action id and record the move.

        Args:
            action_id: RLCard action id.
            chat_text: Optional table talk text for the move.

        Returns:
            Recorded move entry.
        """

        # STEP 1: Resolve action metadata before stepping.
        current_player = self.get_active_player_id()
        action_text = self._formatter.action_id_to_text(action_id)
        player_label = self._player_ids[current_player]

        # STEP 2: Apply the action in RLCard.
        self._state, self._active_player_id = self._env.step(action_id)

        # STEP 3: Record the move.
        move = CardGameMove(
            index=len(self._move_log) + 1,
            player=player_label,
            action_text=action_text,
            raw_action_id=action_id,
            chat_text=chat_text,
        )
        self._move_log.append(move)
        self._last_action = move
        return move

    def parse_action(self, action_text: str, *, player_id: int) -> CardActionParse:
        """Convert an action string into a RLCard action id.

        Args:
            action_text: Action string from the arena layer.
            player_id: RLCard player index for action validation.

        Returns:
            Parsed action data.
        """

        # STEP 1: Normalize incoming text.
        normalized, error = self._formatter.normalize_action_text(action_text)
        if error or not normalized:
            return CardActionParse(None, normalized or "", error or "invalid_action")

        # STEP 2: Validate against legal actions.
        state = self._env.get_state(player_id)
        legal_moves = self._formatter.decode_legal_actions(state.get("raw_legal_actions") or [])
        if normalized not in legal_moves:
            reason = f"illegal_move: '{normalized}' not in legal moves"
            return CardActionParse(None, normalized, reason)

        # STEP 3: Map to action id.
        action_id = self._formatter.action_text_to_id(normalized)
        if action_id is None:
            return CardActionParse(None, normalized, f"unknown_action: '{normalized}'")

        return CardActionParse(int(action_id), normalized, None)

    def build_observation(
        self,
        player_id: int,
        *,
        extra_payload: Optional[dict[str, Any]] = None,
    ) -> tuple[CardGameObservation, str]:
        """Build a structured observation and text snapshot for a player.

        Args:
            player_id: RLCard player index.
            extra_payload: Optional JSON-serializable fields to include in the payload.

        Returns:
            Tuple of structured observation and board text.
        """

        # STEP 1: Fetch the RLCard state for the player.
        state = self._env.get_state(player_id)
        raw_obs = state.get("raw_obs") or {}
        raw_legal_actions = state.get("raw_legal_actions") or []

        # STEP 2: Format public/private state and legal moves.
        public_state = self._formatter.format_public_state(raw_obs, player_id)
        private_state = self._formatter.format_private_state(raw_obs, player_id)
        legal_moves = self._formatter.decode_legal_actions(raw_legal_actions)

        # STEP 3: Build text summary and HTML snapshot.
        board_text = self._formatter.build_text_observation(
            public_state=public_state,
            private_state=private_state,
            legal_moves=legal_moves,
            player_id=player_id,
            player_ids=self._player_ids,
            player_names=self._player_names,
        )
        payload = {
            "game_type": self.game_type,
            "public_state": public_state,
            "private_state": private_state,
            "legal_moves": legal_moves,
            "player_id": player_id,
            "active_player_id": self.get_active_player_id(),
            "player_ids": list(self._player_ids),
            "player_names": dict(self._player_names),
        }
        if extra_payload:
            payload.update(extra_payload)
        board_text = (
            f"{board_text}\n\nSTATE_JSON:\n{json.dumps(payload, ensure_ascii=True)}"
        )
        board_html = self._renderer.render_snapshot(payload, player_ids=self._player_ids, player_names=self._player_names)

        return (
            CardGameObservation(
                board_html=board_html,
                legal_moves_text=legal_moves,
                public_state=public_state,
                private_state=private_state,
            ),
            board_text,
        )

    def move_log(self) -> Sequence[CardGameMove]:
        """Return the recorded move log."""

        return list(self._move_log)

    def last_action(self) -> Optional[CardGameMove]:
        """Return the most recent move."""

        return self._last_action

    def get_payoffs(self) -> Sequence[float]:
        """Return the RLCard payoff list."""

        try:
            payoffs = self._env.get_payoffs()
        except Exception:
            return []
        if hasattr(payoffs, "tolist"):
            return list(payoffs.tolist())
        if isinstance(payoffs, (list, tuple)):
            return list(payoffs)
        return [float(payoffs)]

    def get_all_hands(self) -> dict[str, list[str]]:
        """Return decoded hands for all players when available."""

        game = getattr(self._env, "game", None)
        players = getattr(game, "players", None) if game is not None else None
        if not players:
            return {}

        hands: dict[str, list[str]] = {}
        for idx, player in enumerate(players):
            current_hand = getattr(player, "current_hand", None)
            if current_hand is None:
                current_hand = getattr(player, "_current_hand", None)
            if current_hand is None:
                current_hand = getattr(player, "hand", None)
            if current_hand is None:
                continue
            player_id = self._player_ids[idx] if idx < len(self._player_ids) else str(idx)
            try:
                hands[player_id] = self._formatter.decode_hand(current_hand)
            except Exception:
                if isinstance(current_hand, Sequence):
                    hands[player_id] = [str(card) for card in current_hand]
                else:
                    hands[player_id] = [str(current_hand)]
        return hands
