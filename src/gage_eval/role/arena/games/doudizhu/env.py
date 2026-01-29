"""Generic card arena environment that wires core, formatter, parser, renderer."""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Mapping, Optional, Sequence, Tuple

from loguru import logger

from gage_eval.registry import registry
from gage_eval.role.arena.games.doudizhu.core_factory import make_core
from gage_eval.role.arena.games.doudizhu.cores.base import AbstractGameCore
from gage_eval.role.arena.games.doudizhu.formatters.base import CardGameFormatter
from gage_eval.role.arena.games.doudizhu.formatters.doudizhu import DoudizhuFormatter
from gage_eval.role.arena.games.doudizhu.parsers.base import CardMoveParser, ParsedAction
from gage_eval.role.arena.games.doudizhu.parsers.doudizhu import DoudizhuMoveParser
from gage_eval.role.arena.games.doudizhu.renderers.base import CardGameRenderer
from gage_eval.role.arena.games.doudizhu.renderers.doudizhu import DoudizhuRenderer
from gage_eval.role.arena.games.doudizhu.types import (
    CardChatMessage,
    CardGameAction,
    CardGameObservation,
    CardGameResult,
    CardGameMove,
)
from gage_eval.role.arena.types import ArenaAction, ArenaObservation, GameResult


@registry.asset(
    "arena_impls",
    "doudizhu_rlcard_v1",
    desc="RLCard Doudizhu arena environment",
    tags=("doudizhu", "arena", "card"),
)
class GenericCardArena:
    """Stage manager that coordinates card game components."""

    def __init__(
        self,
        *,
        game_type: str = "doudizhu",
        core: Optional[AbstractGameCore] = None,
        formatter: Optional[CardGameFormatter] = None,
        parser: Optional[CardMoveParser] = None,
        renderer: Optional[CardGameRenderer] = None,
        core_config: Optional[dict[str, Any]] = None,
        player_ids: Optional[Sequence[str]] = None,
        reveal_all_hands: bool = False,
        seat_mode: str = "fixed",
        primary_player_id: Optional[str] = None,
        illegal_action_policy: str = "reject",
        chat_mode: str = "off",
        ai_persona: Optional[dict[str, Any]] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        """Initialize the arena and wire all components.

        Args:
            game_type: Game identifier used by the core factory.
            core: Optional core instance override.
            formatter: Optional formatter instance override.
            parser: Optional parser instance override.
            renderer: Optional renderer instance override.
            core_config: Optional configuration for the core factory.
            player_ids: Optional list of external player ids.
            reveal_all_hands: Whether to expose all hands in observations.
            seat_mode: Seat layout mode ("fixed" or "relative").
            primary_player_id: Anchor player id when using relative seating.
            illegal_action_policy: Policy for illegal actions ("reject", "auto-pass", "random-legal").
            chat_mode: Chat logging mode ("off", "ai-only", "all").
            ai_persona: Optional persona payload for AI players.
            rng_seed: Optional seed for random legal actions.
        """

        self._game_type = str(game_type)
        self._core = core or make_core(self._game_type, config=core_config)
        self._player_ids = list(player_ids or self._default_player_ids(self._core, self._game_type))
        self._player_id_map = {idx: player_id for idx, player_id in enumerate(self._player_ids)}
        self._player_index_map = {player_id: idx for idx, player_id in self._player_id_map.items()}
        self._formatter = formatter or DoudizhuFormatter(player_id_map=self._player_id_map)
        self._parser = parser or DoudizhuMoveParser()
        self._renderer = renderer or DoudizhuRenderer()
        self._reveal_all_hands = bool(reveal_all_hands)
        self._seat_mode = str(seat_mode)
        self._primary_player_id = primary_player_id
        self._illegal_action_policy = str(illegal_action_policy)
        self._chat_mode = str(chat_mode)
        self._ai_persona = dict(ai_persona or {"enabled": False, "prompt": ""})
        self._rng = random.Random(rng_seed)
        self._move_log: list[CardGameMove] = []
        self._chat_log: list[CardChatMessage] = []
        self._final_result: Optional[CardGameResult] = None
        self._last_frame: Optional[dict[str, Any] | str] = None

    def reset(self) -> None:
        """Reset the arena state and underlying core."""

        self._core.reset()
        self._move_log = []
        self._chat_log = []
        self._final_result = None
        self._last_frame = None

    def get_active_player(self) -> str:
        """Return the external player id for the active player.

        Returns:
            External player identifier.
        """

        return self._player_id_map[self._core.get_active_player_id()]

    def observe(self, player_id: str) -> CardGameObservation:
        """Return a formatted observation for a player.

        Args:
            player_id: External player identifier.

        Returns:
            Formatted observation payload.
        """

        # STEP 1: Resolve raw observation and legal actions.
        player_index = self._resolve_player_index(player_id)
        raw_obs = self._core.get_observation(player_index)
        legal_action_ids = self._core.get_legal_actions(player_index)

        # STEP 2: Format public/private state and legal moves.
        public_state, private_state, legal_moves = self._formatter.format_observation(
            raw_obs,
            legal_action_ids,
        )
        public_state.update(
            {
                "seat_mode": self._seat_mode,
                "primary_player_id": self._primary_player_id,
            }
        )
        if self._reveal_all_hands:
            hands = self._core.get_all_hands()
            if hands is not None:
                public_state["all_hands"] = self._map_hands(hands)

        # STEP 3: Render and build the observation payload.
        frame = {
            "active_player_id": self.get_active_player(),
            "public_state": public_state,
            "private_state": private_state,
        }
        self._last_frame = self._renderer.render_frame(frame)
        chat_log = [] if self._chat_mode == "off" else list(self._chat_log)
        return CardGameObservation(
            public_state=public_state,
            private_state=private_state,
            legal_moves=legal_moves,
            player_id=player_id,
            active_player_id=self.get_active_player(),
            chat_log=chat_log,
            ai_persona=dict(self._ai_persona),
        )

    def apply(self, action: CardGameAction | str) -> Optional[CardGameResult]:
        """Apply an action and return the GameResult if the game ends.

        Args:
            action: Player action payload or raw text.

        Returns:
            Final game result if the game terminates, otherwise None.
        """

        if self._final_result is not None:
            return self._final_result

        # STEP 1: Resolve the acting player and legal actions.
        player_id = self._resolve_action_player_id(action)
        if player_id != self.get_active_player():
            return self._handle_illegal(player_id, reason="wrong_player")

        player_index = self._resolve_player_index(player_id)
        legal_action_ids = self._core.get_legal_actions(player_index)

        # STEP 2: Parse and validate the action payload.
        try:
            parsed = self._parser.parse(self._action_payload(action))
        except Exception as exc:
            return self._handle_illegal(player_id, reason=f"action_parse_error:{exc}")

        # STEP 3: Apply the action or fallback according to policy.
        if parsed.action_id not in legal_action_ids:
            return self._handle_illegal(player_id, reason="illegal_move", legal_action_ids=legal_action_ids)

        self._record_chat(player_id, parsed.chat_text)
        return self._execute_action(player_id, parsed, legal_action_ids)

    def is_terminal(self) -> bool:
        """Return True if the game has ended.

        Returns:
            True when the game has reached a terminal state.
        """

        return self._final_result is not None or self._core.is_terminal()

    def build_result(self, *, reason: str) -> CardGameResult:
        """Build a result snapshot for the completed game.

        Args:
            reason: Terminal reason string.

        Returns:
            Game result payload.
        """

        # STEP 1: Short-circuit if result is already finalized.
        if self._final_result is not None:
            return self._final_result

        # STEP 2: Resolve payoffs and winner.
        payoffs = self._safe_payoffs()
        winner = self._resolve_winner(payoffs)
        result: CardGameResult = {
            "winner": winner,
            "reason": reason,
            "final_board_html": "",
            "move_log": list(self._move_log),
            "chat_log": list(self._chat_log),
            "payoffs": payoffs,
        }

        # STEP 3: Render the replay payload.
        replay = self._renderer.save_replay(result)
        result["final_board_html"] = self._stringify_replay(replay)
        self._final_result = result
        return result

    def _execute_action(
        self,
        player_id: str,
        parsed: ParsedAction,
        legal_action_ids: Sequence[int],
    ) -> Optional[CardGameResult]:
        action_text = self._formatter.format_action(parsed.action_id)
        self._move_log.append(
            {
                "player_id": player_id,
                "action_id": int(parsed.action_id),
                "action_text": action_text,
                "chat": parsed.chat_text,
            }
        )
        self._core.step(parsed.action_id)
        if self._core.is_terminal():
            return self.build_result(reason="terminal")
        return None

    def _handle_illegal(
        self,
        player_id: str,
        *,
        reason: str,
        legal_action_ids: Optional[Sequence[int]] = None,
    ) -> Optional[CardGameResult]:
        policy = self._illegal_action_policy
        if policy == "auto-pass":
            fallback = self._resolve_pass_action(legal_action_ids or [])
            if fallback is not None:
                parsed = ParsedAction(action_id=fallback, action_text="pass")
                return self._execute_action(player_id, parsed, legal_action_ids or [])
        if policy == "random-legal":
            fallback = self._resolve_random_action(legal_action_ids or [])
            if fallback is not None:
                parsed = ParsedAction(action_id=fallback, action_text=self._formatter.format_action(fallback))
                return self._execute_action(player_id, parsed, legal_action_ids or [])
        return self.build_result(reason=reason)

    def _record_chat(self, player_id: str, chat_text: Optional[str]) -> None:
        if not chat_text or self._chat_mode == "off":
            return
        self._chat_log.append({"player_id": player_id, "text": chat_text})

    def _resolve_action_player_id(self, action: CardGameAction | str) -> str:
        if isinstance(action, Mapping):
            player_id = action.get("player_id")
            if player_id:
                return str(player_id)
        return self.get_active_player()

    def _action_payload(self, action: CardGameAction | str) -> str | Mapping[str, Any]:
        if isinstance(action, Mapping):
            return action
        return str(action)

    def _resolve_player_index(self, player_id: str) -> int:
        if player_id not in self._player_index_map:
            raise KeyError(f"Unknown player id: {player_id}")
        return int(self._player_index_map[player_id])

    def _resolve_pass_action(self, legal_action_ids: Sequence[int]) -> Optional[int]:
        if not legal_action_ids:
            return None
        try:
            pass_id = self._core.encode_action("pass")
        except Exception:
            pass_id = None
        if pass_id is not None and pass_id in legal_action_ids:
            return int(pass_id)
        return self._resolve_random_action(legal_action_ids)

    def _resolve_random_action(self, legal_action_ids: Sequence[int]) -> Optional[int]:
        if not legal_action_ids:
            return None
        return int(self._rng.choice(list(legal_action_ids)))

    def _safe_payoffs(self) -> list[float]:
        try:
            if self._core.is_terminal():
                return self._core.get_payoffs()
        except Exception:
            pass
        return [0.0 for _ in self._player_ids]

    def _resolve_winner(self, payoffs: Sequence[float]) -> Optional[str]:
        if not payoffs:
            return None
        max_payoff = max(payoffs)
        if max_payoff <= 0:
            return None
        winner_index = list(payoffs).index(max_payoff)
        return self._player_id_map.get(winner_index)

    def _stringify_replay(self, replay: dict[str, Any] | str) -> str:
        if isinstance(replay, str):
            return replay
        return json.dumps(replay, ensure_ascii=True)

    def _map_hands(self, hands: Mapping[str, str]) -> dict[str, str]:
        mapped: dict[str, str] = {}
        for player_key, hand in hands.items():
            try:
                player_index = int(player_key)
            except (TypeError, ValueError):
                mapped[player_key] = hand
                continue
            mapped[self._player_id_map.get(player_index, player_key)] = hand
        return mapped

    def _default_player_ids(self, core: AbstractGameCore, game_type: str) -> list[str]:
        num_players = int(getattr(core, "num_players", 0) or 0)
        if num_players <= 0:
            num_players = 3 if game_type == "doudizhu" else 2
        return [f"player_{idx}" for idx in range(num_players)]


@registry.asset(
    "arena_impls",
    "doudizhu_arena_v1",
    desc="Arena environment adapter for RLCard Doudizhu",
    tags=("doudizhu", "arena"),
)
class DoudizhuArenaEnvironment:
    """Arena environment that adapts RLCard Doudizhu to ArenaRoleAdapter."""

    def __init__(
        self,
        *,
        board_size: int = 0,
        win_len: int = 0,
        player_ids: Optional[Sequence[str]] = None,
        player_names: Optional[dict[str, str]] = None,
        token_map: Optional[dict[str, str]] = None,
        start_player_id: Optional[str] = None,
        coord_scheme: str = "A1",
        rule_profile: Optional[str] = None,
        win_directions: Optional[Sequence[str]] = None,
        illegal_policy: Optional[dict[str, str | int]] = None,
        chat_mode: str = "off",
        chat_every_n: int = 1,
        context_include_public: bool = True,
        context_include_ui_state: bool = True,
        run_id: Optional[str] = None,
        sample_id: Optional[str] = None,
        replay_output_dir: Optional[str] = None,
        replay_filename: Optional[str] = None,
        replay_live: bool = False,
        chat_queue: Optional[Queue[dict[str, str] | str]] = None,
        fast_finish_action: Optional[str] = None,
        fast_finish_human_only: bool = True,
        **_: Any,
    ) -> None:
        """Initialize the Doudizhu arena environment.

        Args:
            board_size: Unused placeholder for ArenaRoleAdapter compatibility.
            win_len: Unused placeholder for ArenaRoleAdapter compatibility.
            player_ids: Optional list of player identifiers.
            player_names: Optional mapping from player id to display name.
            token_map: Unused placeholder for ArenaRoleAdapter compatibility.
            start_player_id: Optional starting player identifier.
            coord_scheme: Unused placeholder for ArenaRoleAdapter compatibility.
            rule_profile: Optional rule profile label.
            win_directions: Unused placeholder for ArenaRoleAdapter compatibility.
            illegal_policy: Policy controlling illegal move retries and outcome.
            chat_mode: Chat logging mode ("off", "ai-only", "all").
            chat_every_n: Record chat every N moves (>=1).
            context_include_public: Whether to include public_state in AI context.
            context_include_ui_state: Whether to include UI_STATE_JSON in AI context.
            run_id: Optional run identifier used for replay output.
            sample_id: Optional sample identifier used for replay output.
            replay_output_dir: Optional override for the replay output directory.
            replay_filename: Optional override for the replay filename.
            replay_live: Whether to update the replay file after each move.
            chat_queue: Optional queue for chat-only messages.
            fast_finish_action: Optional action string to force a fast finish.
            fast_finish_human_only: Whether to allow fast finish only for human players.
            **_: Ignored extra keyword arguments.
        """

        _ = (board_size, win_len, token_map, coord_scheme, win_directions)
        self._core = make_core("doudizhu")
        resolved_ids = list(player_ids or self._default_player_ids())
        self._player_ids = [str(player_id) for player_id in resolved_ids]
        self._player_names = dict(player_names or {})
        for player_id in self._player_ids:
            self._player_names.setdefault(player_id, player_id)
        self._player_id_map = {idx: player_id for idx, player_id in enumerate(self._player_ids)}
        self._player_index_map = {player_id: idx for idx, player_id in self._player_id_map.items()}
        self._start_player_id = (
            str(start_player_id) if start_player_id in self._player_ids else self._player_ids[0]
        )
        self._formatter = DoudizhuFormatter(player_id_map=self._player_id_map)
        self._rule_profile = str(rule_profile or "doudizhu")
        self._illegal_policy = dict(illegal_policy or {})
        self._max_illegal = int(self._illegal_policy.get("retry", 0))
        self._illegal_on_fail = str(self._illegal_policy.get("on_fail", "loss"))
        self._chat_mode = str(chat_mode or "off")
        self._chat_every_n = max(1, int(chat_every_n))
        self._context_include_public = bool(context_include_public)
        self._context_include_ui_state = bool(context_include_ui_state)
        self._run_id = str(run_id) if run_id else None
        self._sample_id = str(sample_id) if sample_id else None
        self._replay_output_dir = str(replay_output_dir) if replay_output_dir else None
        self._replay_filename = str(replay_filename) if replay_filename else None
        self._replay_live = bool(replay_live)
        self._chat_queue = chat_queue
        self._fast_finish_action = str(fast_finish_action).strip().lower() if fast_finish_action else None
        self._fast_finish_human_only = bool(fast_finish_human_only)
        self._replay_path: Optional[str] = None
        self._illegal_counts = {player_id: 0 for player_id in self._player_ids}
        self._move_log: list[dict[str, Any]] = []
        self._move_history: list[dict[str, Any]] = []
        self._chat_log: list[dict[str, str]] = []
        self._move_count = 0
        self._illegal_move_count = 0
        self._last_move: Optional[str] = None
        self._hand_cards_with_suit: list[list[str]] = []
        self._initial_hands: list[list[str]] = []
        self._latest_actions: list[list[str] | str] = []
        self._landlord_id: Optional[str] = None
        self._final_result: Optional[GameResult] = None
        self._start_time_ms: Optional[int] = None
        self.reset()

    def reset(self) -> None:
        """Reset the arena and core state."""

        self._core.reset()
        self._illegal_counts = {player_id: 0 for player_id in self._player_ids}
        self._move_log = []
        self._move_history = []
        self._chat_log = []
        self._move_count = 0
        self._illegal_move_count = 0
        self._last_move = None
        self._latest_actions = [[] for _ in self._player_ids]
        self._hand_cards_with_suit = []
        self._initial_hands = []
        self._landlord_id = None
        self._replay_path = None
        self._final_result = None
        self._start_time_ms = int(time.time() * 1000)
        self._drain_chat_queue(record=False)
        self._refresh_perfect_information()
        self._initial_hands = [list(cards) for cards in self._hand_cards_with_suit]
        if self._replay_live:
            self._save_showdown_replay()

    def get_active_player(self) -> str:
        """Return the active player identifier."""

        return self._player_id_map[self._core.get_active_player_id()]

    def observe(self, player: str) -> ArenaObservation:
        """Return an observation for the given player."""

        self._drain_chat_queue()
        player_index = self._resolve_player_index(player)
        raw_obs = self._core.get_observation(player_index)
        legal_action_ids = self._core.get_legal_actions(player_index)
        self._refresh_perfect_information()
        public_state, private_state, legal_moves = self._formatter.format_observation(
            raw_obs,
            legal_action_ids,
        )
        ui_state = self._build_ui_state(public_state, private_state, legal_moves)
        board_text = self._format_board_text(
            public_state,
            private_state,
            legal_moves,
            chat_log=self._chat_log,
            ui_state=ui_state if self._context_include_ui_state else None,
            include_public_state=self._context_include_public,
            include_private_state=True,
        )
        view = {"text": board_text}
        legal_actions = {"items": list(legal_moves)}
        context = {"mode": "turn", "step": self._move_count}
        return ArenaObservation(
            board_text=board_text,
            legal_moves=legal_moves,
            active_player=self.get_active_player(),
            last_move=self._last_move,
            metadata={
                "player_id": player,
                "player_ids": list(self._player_ids),
                "player_names": dict(self._player_names),
                "start_player_id": self._start_player_id,
                "rule_profile": self._rule_profile,
                "public_state": public_state,
                "private_state": private_state,
                "chat_log": list(self._chat_log),
                "chat_mode": self._chat_mode,
                "last_move": self._last_move,
            },
            view=view,
            legal_actions=legal_actions,
            context=context,
        )

    def _drain_chat_queue(self, *, record: bool = True) -> None:
        if self._chat_queue is None:
            return
        added = False
        while True:
            try:
                payload = self._chat_queue.get_nowait()
            except Empty:
                break
            parsed = self._parse_chat_payload(payload)
            if not parsed:
                continue
            player_id, chat_text = parsed
            if record:
                self._record_chat(player_id, chat_text, "human")
                added = True
        if added and self._replay_live:
            self._save_showdown_replay()

    def _parse_chat_payload(self, payload: Any) -> Optional[Tuple[str, str]]:
        if payload is None:
            return None
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return None
            if text.startswith("{"):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    return (self._resolve_chat_player_id(None), text)
                if isinstance(parsed, dict):
                    chat_text = parsed.get("chat") or parsed.get("text") or parsed.get("message")
                    player_id = parsed.get("player_id") or parsed.get("playerId") or parsed.get("player")
                    if chat_text:
                        return (self._resolve_chat_player_id(player_id), str(chat_text).strip())
            return (self._resolve_chat_player_id(None), text)
        if isinstance(payload, dict):
            chat_text = payload.get("chat") or payload.get("text") or payload.get("message")
            if not chat_text:
                return None
            player_id = payload.get("player_id") or payload.get("playerId") or payload.get("player")
            return (self._resolve_chat_player_id(player_id), str(chat_text).strip())
        return None

    def _resolve_chat_player_id(self, player_id: Optional[object]) -> str:
        if player_id is not None:
            if isinstance(player_id, int):
                if player_id in self._player_id_map:
                    return self._player_id_map[player_id]
            else:
                raw = str(player_id).strip()
                if raw.isdigit():
                    parsed_idx = int(raw)
                    if parsed_idx in self._player_id_map:
                        return self._player_id_map[parsed_idx]
                if raw in self._player_index_map:
                    return raw
        if self._start_player_id in self._player_index_map:
            return self._start_player_id
        return self._player_ids[0]

    def apply(self, action: ArenaAction) -> Optional[GameResult]:
        """Apply an action and return a GameResult when the game ends."""

        if self._final_result is not None:
            return self._final_result

        player_id = action.player
        if player_id != self.get_active_player():
            return self._handle_illegal(player_id, reason="wrong_player")

        action_text = action.move or ""
        raw_action = str(action.raw or "")
        if self._should_fast_finish(action_text, raw_action, action.metadata.get("player_type")):
            return self._apply_fast_finish(player_id, action)
        if not action_text:
            return self._handle_illegal(player_id, reason="empty_move")

        player_index = self._resolve_player_index(player_id)
        legal_action_ids = self._core.get_legal_actions(player_index)
        try:
            action_id = self._core.encode_action(action_text)
        except Exception:
            return self._handle_illegal(player_id, reason="invalid_action")

        if action_id not in legal_action_ids:
            return self._handle_illegal(player_id, reason="illegal_move")

        # STEP 1: Apply the legal action and update logs.
        self._refresh_perfect_information()
        self._move_count += 1
        action_text = self._formatter.format_action(action_id)
        timestamp_ms = int(time.time() * 1000)
        action_cards = self._resolve_action_cards(action_text, player_index)
        chat_text = self._extract_chat_text(action)
        self._record_chat(player_id, chat_text, action.metadata.get("player_type"))
        self._latest_actions[player_index] = action_cards
        self._move_history.append(
            {
                "index": self._move_count,
                "player_id": player_id,
                "player_idx": player_index,
                "action_text": action_text,
                "action_cards": action_cards,
                "chat": chat_text,
                "timestamp_ms": timestamp_ms,
            }
        )
        self._move_log.append(
            {
                "index": self._move_count,
                "player": player_id,
                "action_id": int(action_id),
                "action_text": action_text,
                "action_cards": action_cards,
                "chat": chat_text,
                "timestamp_ms": timestamp_ms,
            }
        )
        self._last_move = action_text
        self._core.step(action_id)
        self._refresh_perfect_information()
        if self._replay_live:
            self._save_showdown_replay()

        # STEP 2: Check terminal state.
        if self._core.is_terminal():
            return self.build_result(result="win", reason="terminal")
        return None

    def is_terminal(self) -> bool:
        """Return True if the game has ended."""

        return self._final_result is not None or self._core.is_terminal()

    def build_result(self, *, result: str, reason: Optional[str]) -> GameResult:
        """Build a final GameResult snapshot."""

        if self._final_result is not None:
            return self._final_result

        payoffs = self._safe_payoffs()
        winner = self._resolve_winner(payoffs)
        resolved_result = "draw" if result == "draw" or winner is None else result
        final_board = self._snapshot_board()
        replay_path = self._save_showdown_replay()
        self._final_result = GameResult(
            winner=winner,
            result=resolved_result,
            reason=reason,
            move_count=self._move_count,
            illegal_move_count=self._illegal_move_count,
            final_board=final_board,
            move_log=list(self._move_log),
            rule_profile=self._rule_profile,
            win_direction=None,
            line_length=None,
            replay_path=replay_path,
        )
        return self._final_result

    def _handle_illegal(self, player_id: str, *, reason: str) -> Optional[GameResult]:
        if player_id in self._illegal_counts:
            self._illegal_counts[player_id] += 1
        self._illegal_move_count += 1
        if self._max_illegal < 0:
            return None
        if self._illegal_counts.get(player_id, 0) <= self._max_illegal:
            return None

        if self._illegal_on_fail == "draw":
            winner = None
            result = "draw"
        else:
            winner = self._resolve_illegal_winner(player_id)
            result = "loss"

        replay_path = self._save_showdown_replay()
        self._final_result = GameResult(
            winner=winner,
            result=result,
            reason=reason,
            move_count=self._move_count,
            illegal_move_count=self._illegal_move_count,
            final_board=self._snapshot_board(),
            move_log=list(self._move_log),
            rule_profile=self._rule_profile,
            win_direction=None,
            line_length=None,
            replay_path=replay_path,
        )
        return self._final_result

    def _resolve_player_index(self, player_id: str) -> int:
        if player_id not in self._player_index_map:
            raise KeyError(f"Unknown player id: {player_id}")
        return int(self._player_index_map[player_id])

    def _resolve_illegal_winner(self, offender_id: str) -> Optional[str]:
        for player_id in self._player_ids:
            if player_id != offender_id:
                return player_id
        return None

    def _safe_payoffs(self) -> list[float]:
        try:
            if self._core.is_terminal():
                return self._core.get_payoffs()
        except Exception:
            pass
        return [0.0 for _ in self._player_ids]

    def _resolve_winner(self, payoffs: Sequence[float]) -> Optional[str]:
        if not payoffs:
            return None
        max_payoff = max(payoffs)
        if max_payoff <= 0:
            return None
        winner_index = list(payoffs).index(max_payoff)
        return self._player_id_map.get(winner_index)

    def _save_showdown_replay(self) -> Optional[str]:
        try:
            replay = self._build_showdown_replay()
        except Exception as exc:
            logger.warning("Failed to build showdown replay: {}", exc)
            return None

        output_path = self._resolve_replay_output_path()
        if output_path is None:
            return None

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8")
            self._replay_path = str(output_path)
            return self._replay_path
        except Exception as exc:
            logger.warning("Failed to write showdown replay to {}: {}", output_path, exc)
            return None

    def _build_showdown_replay(self) -> dict[str, Any]:
        self._ensure_landlord_id()
        active_index = self._core.get_active_player_id()
        try:
            legal_action_ids = self._core.get_legal_actions(active_index)
        except Exception:
            legal_action_ids = []
        legal_moves = [self._formatter.format_action(action_id) for action_id in legal_action_ids]
        legal_moves = list(dict.fromkeys(legal_moves))
        roles = self._resolve_roles(self._landlord_id)
        player_info = []
        for idx, player_id in enumerate(self._player_ids):
            player_info.append(
                {
                    "id": idx,
                    "index": idx,
                    "role": roles.get(player_id, "peasant"),
                    "agentInfo": {"name": self._player_names.get(player_id, player_id)},
                }
            )

        init_hands = []
        if self._initial_hands:
            init_hands = [" ".join(cards) for cards in self._initial_hands]
        else:
            init_hands = [" ".join(cards) for cards in self._hand_cards_with_suit]
        if len(init_hands) < len(self._player_ids):
            init_hands.extend(["" for _ in range(len(self._player_ids) - len(init_hands))])
        else:
            init_hands = init_hands[: len(self._player_ids)]

        move_history = []
        for entry in self._move_history:
            player_idx = entry.get("player_idx")
            action_cards = entry.get("action_cards")
            action_text = entry.get("action_text")
            move = action_cards if action_cards not in (None, "") else action_text
            if isinstance(move, list):
                move_text = " ".join(move)
            elif move is None:
                move_text = ""
            else:
                move_text = str(move)
            move_history.append(
                {
                    "playerIdx": int(player_idx) if player_idx is not None else 0,
                    "move": move_text,
                    "info": {},
                    "chat": entry.get("chat"),
                    "timestamp_ms": entry.get("timestamp_ms"),
                }
            )

        return {
            "playerInfo": player_info,
            "initHands": init_hands,
            "moveHistory": move_history,
            "chatLog": list(self._chat_log),
            "currentPlayer": int(active_index),
            "active_player_id": self._player_id_map.get(active_index),
            "legalMoves": legal_moves,
            "start_time_ms": self._start_time_ms,
        }

    def _ensure_landlord_id(self) -> None:
        if self._landlord_id:
            return
        try:
            raw_obs = self._core.get_observation(0)
            legal_action_ids = self._core.get_legal_actions(0)
            public_state, _, _ = self._formatter.format_observation(raw_obs, legal_action_ids)
            landlord_id = public_state.get("landlord_id")
            if landlord_id and landlord_id != "unknown":
                self._landlord_id = str(landlord_id)
        except Exception:
            return

    def _resolve_replay_output_path(self) -> Optional[Path]:
        base_dir = None
        if self._replay_output_dir:
            base_dir = Path(self._replay_output_dir)
        else:
            run_id = self._run_id or os.environ.get("GAGE_EVAL_RUN_ID")
            if run_id:
                base_dir = Path(os.environ.get("GAGE_EVAL_SAVE_DIR", "./runs")) / run_id / "replays"
        if base_dir is None:
            return None
        sample_id_source = self._sample_id or os.environ.get("GAGE_EVAL_SAMPLE_ID") or "unknown"
        sample_id = self._sanitize_sample_id(sample_id_source)
        filename = self._replay_filename or f"doudizhu_replay_{sample_id}.json"
        return base_dir / filename

    def _sanitize_sample_id(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_") or "unknown"

    def _snapshot_board(self) -> str:
        try:
            self._refresh_perfect_information()
            raw_obs = self._core.get_observation(0)
            legal_action_ids = self._core.get_legal_actions(0)
            public_state, private_state, legal_moves = self._formatter.format_observation(
                raw_obs,
                legal_action_ids,
            )
            ui_state = self._build_ui_state(public_state, private_state, legal_moves)
            return self._format_board_text(
                public_state,
                private_state,
                legal_moves,
                chat_log=self._chat_log,
                ui_state=ui_state,
            )
        except Exception:
            return ""

    @staticmethod
    def _format_board_text(
        public_state: dict[str, Any],
        private_state: dict[str, Any],
        legal_moves: Sequence[str],
        *,
        chat_log: Optional[Sequence[dict[str, str]]] = None,
        ui_state: Optional[dict[str, Any]] = None,
        include_public_state: bool = True,
        include_private_state: bool = True,
        include_ui_state: bool = True,
    ) -> str:
        legal_preview = ", ".join(list(legal_moves)[:40]) if legal_moves else "none"
        lines: list[str] = []
        if include_public_state:
            lines.extend(
                [
                    "Public State:",
                    json.dumps(public_state, ensure_ascii=True),
                    "",
                ]
            )
        if include_private_state:
            lines.extend(
                [
                    "Private State:",
                    json.dumps(private_state, ensure_ascii=True),
                    "",
                ]
            )
        lines.append(f"Legal Moves (preview): {legal_preview}")
        if chat_log is not None:
            lines.extend(
                [
                    "",
                    "Chat Log:",
                    json.dumps(list(chat_log), ensure_ascii=True),
                ]
            )
        if include_ui_state and ui_state is not None:
            lines.extend(
                [
                    "",
                    "UI_STATE_JSON:",
                    json.dumps(ui_state, ensure_ascii=True),
                ]
            )
        return "\n".join(lines)

    def _refresh_perfect_information(self) -> dict[str, Any]:
        perfect = self._core.get_perfect_information()
        if not isinstance(perfect, dict):
            return {}
        hand_cards = self._parse_hand_cards_with_suit(perfect.get("hand_cards_with_suit", []))
        self._hand_cards_with_suit = hand_cards
        return perfect

    def _parse_hand_cards_with_suit(self, hand_cards: Sequence[Any]) -> list[list[str]]:
        parsed: list[list[str]] = []
        for entry in list(hand_cards):
            if entry is None:
                parsed.append([])
                continue
            if isinstance(entry, str):
                cards = [card for card in entry.split() if card]
            else:
                try:
                    cards = [str(card) for card in list(entry)]
                except TypeError:
                    cards = [str(entry)]
            parsed.append(cards)
        return parsed

    def _build_ui_state(
        self,
        public_state: dict[str, Any],
        private_state: dict[str, Any],
        legal_moves: Sequence[str],
    ) -> dict[str, Any]:
        _ = (private_state, legal_moves)
        landlord_id = public_state.get("landlord_id")
        if landlord_id and landlord_id != "unknown":
            self._landlord_id = str(landlord_id)
        roles = self._resolve_roles(self._landlord_id)
        seat_order = self._resolve_seat_order(self._start_player_id)
        hand_cards = list(self._hand_cards_with_suit or [])
        if len(hand_cards) < len(self._player_ids):
            hand_cards.extend([[] for _ in range(len(self._player_ids) - len(hand_cards))])
        else:
            hand_cards = hand_cards[: len(self._player_ids)]
        latest_actions = list(self._latest_actions or [])
        if len(latest_actions) < len(self._player_ids):
            latest_actions.extend([[] for _ in range(len(self._player_ids) - len(latest_actions))])
        else:
            latest_actions = latest_actions[: len(self._player_ids)]
        return {
            "player_ids": list(self._player_ids),
            "player_names": dict(self._player_names),
            "roles": roles,
            "seat_order": seat_order,
            "hands": hand_cards,
            "latest_actions": latest_actions,
            "active_player_id": self.get_active_player(),
            "landlord_id": self._landlord_id or "unknown",
            "move_count": self._move_count,
            "move_history": list(self._move_history),
            "chat_log": list(self._chat_log),
        }

    def _resolve_roles(self, landlord_id: Optional[str]) -> dict[str, str]:
        roles: dict[str, str] = {}
        for player_id in self._player_ids:
            roles[player_id] = "landlord" if landlord_id and player_id == landlord_id else "peasant"
        return roles

    def _resolve_seat_order(self, anchor_id: str) -> dict[str, str]:
        if not self._player_ids:
            return {}
        anchor = anchor_id if anchor_id in self._player_ids else self._player_ids[0]
        anchor_index = self._player_ids.index(anchor)
        total = len(self._player_ids)
        left_index = (anchor_index + 1) % total
        right_index = (anchor_index + 2) % total
        return {
            "bottom": self._player_ids[anchor_index],
            "left": self._player_ids[left_index],
            "right": self._player_ids[right_index],
        }

    def _resolve_action_cards(self, action_text: str, player_index: int) -> list[str] | str:
        normalized = str(action_text).strip()
        if not normalized:
            return ""
        if normalized == "pass":
            return "pass"
        if player_index < 0 or player_index >= len(self._hand_cards_with_suit):
            return normalized
        hand_cards = self._hand_cards_with_suit[player_index]
        if not hand_cards:
            return normalized
        trans = {"B": "BJ", "R": "RJ"}
        cards_with_suit: list[str] = []
        for char in normalized:
            if char in trans:
                target = trans[char]
                if target in hand_cards:
                    hand_cards.remove(target)
                cards_with_suit.append(target)
                continue
            matched = None
            for card in hand_cards:
                if len(card) > 1 and card[1] == char:
                    matched = card
                    break
            if matched:
                hand_cards.remove(matched)
                cards_with_suit.append(matched)
        if not cards_with_suit:
            return normalized
        return cards_with_suit

    def _should_fast_finish(
        self,
        action_text: str,
        raw_action: str,
        player_type: Optional[str],
    ) -> bool:
        if not self._fast_finish_action:
            return False
        if self._fast_finish_human_only and player_type != "human":
            return False
        candidate = action_text.strip().lower()
        if not candidate:
            candidate = self._extract_action_from_raw(raw_action).strip().lower()
        return candidate == self._fast_finish_action

    def _extract_action_from_raw(self, raw_action: str) -> str:
        cleaned = str(raw_action or "").strip()
        if not cleaned:
            return ""
        if cleaned.startswith("{"):
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                return cleaned
            if isinstance(parsed, dict):
                action = parsed.get("action")
                return str(action or "").strip()
        return cleaned

    def _apply_fast_finish(self, player_id: str, action: ArenaAction) -> Optional[GameResult]:
        player_index = self._resolve_player_index(player_id)
        self._refresh_perfect_information()
        action_cards = list(self._hand_cards_with_suit[player_index]) if self._hand_cards_with_suit else []
        self._hand_cards_with_suit[player_index] = []
        self._move_count += 1
        action_text = self._fast_finish_action or "finish"
        timestamp_ms = int(time.time() * 1000)
        chat_text = self._extract_chat_text(action)
        self._record_chat(player_id, chat_text, action.metadata.get("player_type"))
        self._latest_actions[player_index] = action_cards
        self._move_history.append(
            {
                "index": self._move_count,
                "player_id": player_id,
                "player_idx": player_index,
                "action_text": action_text,
                "action_cards": action_cards,
                "chat": chat_text,
                "timestamp_ms": timestamp_ms,
            }
        )
        self._move_log.append(
            {
                "index": self._move_count,
                "player": player_id,
                "action_id": -1,
                "action_text": action_text,
                "action_cards": action_cards,
                "chat": chat_text,
                "timestamp_ms": timestamp_ms,
            }
        )
        self._last_move = action_text
        if self._replay_live:
            self._save_showdown_replay()
        self._final_result = GameResult(
            winner=player_id,
            result="win",
            reason="fast_finish",
            move_count=self._move_count,
            illegal_move_count=self._illegal_move_count,
            final_board=self._snapshot_board(),
            move_log=list(self._move_log),
            rule_profile=self._rule_profile,
            win_direction=None,
            line_length=None,
            replay_path=self._replay_path,
        )
        return self._final_result

    def _extract_chat_text(self, action: ArenaAction) -> Optional[str]:
        chat_text = action.metadata.get("chat")
        if chat_text:
            cleaned = str(chat_text).strip()
            if cleaned:
                return cleaned
        raw_text = str(action.raw or "").strip()
        if raw_text.startswith("{"):
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, dict):
                chat_text = parsed.get("chat")
                if chat_text:
                    cleaned = str(chat_text).strip()
                    return cleaned or None
        return None

    def _record_chat(self, player_id: str, chat_text: Optional[str], player_type: Optional[str]) -> None:
        if not chat_text or self._chat_mode == "off":
            return
        if self._chat_mode == "ai-only" and player_type == "human":
            return
        if self._chat_every_n > 1 and (self._move_count - 1) % self._chat_every_n != 0:
            return
        self._chat_log.append({"player_id": player_id, "text": chat_text})

    def _default_player_ids(self) -> list[str]:
        num_players = int(getattr(self._core, "num_players", 0) or 0)
        if num_players <= 0:
            num_players = 3
        return [f"player_{idx}" for idx in range(num_players)]
