"""Generic Mahjong arena environment that wires core, formatter, parser, renderer."""

from __future__ import annotations

import json
import os
import random
import re
import time
from queue import Empty, Queue
from threading import Event, Thread
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from loguru import logger

from gage_eval.registry import registry
from gage_eval.role.arena.games.mahjong.core_factory import make_core
from gage_eval.role.arena.games.mahjong.cores.base import AbstractGameCore
from gage_eval.role.arena.games.mahjong.formatters.base import MahjongFormatter
from gage_eval.role.arena.games.mahjong.formatters.mahjong import StandardMahjongFormatter
from gage_eval.role.arena.games.mahjong.mapping import build_action_maps, is_tile_action
from gage_eval.role.arena.games.mahjong.parsers.base import MahjongMoveParser, MahjongParsedAction
from gage_eval.role.arena.games.mahjong.parsers.mahjong import StandardMahjongParser
from gage_eval.role.arena.games.mahjong.renderers.base import MahjongRenderer
from gage_eval.role.arena.games.mahjong.renderers.mahjong import StandardMahjongRenderer
from gage_eval.role.arena.games.mahjong.types import MahjongAction, MahjongChatMessage, MahjongMove
from gage_eval.role.arena.types import ArenaAction, ArenaObservation, GameResult


@registry.asset(
    "arena_impls",
    "mahjong_rlcard_v1",
    desc="RLCard Mahjong arena environment",
    tags=("mahjong", "arena", "card"),
)
class MahjongArena:
    """Stage manager that coordinates Mahjong game components."""

    def __init__(
        self,
        *,
        game_type: str = "mahjong",
        core: Optional[AbstractGameCore] = None,
        formatter: Optional[MahjongFormatter] = None,
        parser: Optional[MahjongMoveParser] = None,
        renderer: Optional[MahjongRenderer] = None,
        core_config: Optional[dict[str, Any]] = None,
        player_ids: Optional[Sequence[str]] = None,
        player_names: Optional[dict[str, str]] = None,
        seat_mode: str = "fixed",
        illegal_policy: Optional[dict[str, str | int]] = None,
        illegal_action_policy: str = "reject",
        chat_mode: str = "off",
        chat_every_n: int = 1,
        chat_queue: Optional[Queue[dict[str, str]]] = None,
        ai_persona: Optional[dict[str, Any]] = None,
        rng_seed: Optional[int] = None,
        run_id: Optional[str] = None,
        sample_id: Optional[str] = None,
        replay_output_dir: Optional[str] = None,
        replay_filename: Optional[str] = None,
        replay_live: bool = False,
        **_: Any,
    ) -> None:
        """Initialize the Mahjong arena environment.

        Args:
            game_type: Game identifier used by the core factory.
            core: Optional core instance override.
            formatter: Optional formatter instance override.
            parser: Optional parser instance override.
            renderer: Optional renderer instance override.
            core_config: Optional configuration for the core factory.
            player_ids: Optional list of external player ids.
            player_names: Optional mapping from player id to display name.
            seat_mode: Seat layout mode ("fixed" or "relative").
            illegal_policy: Policy controlling illegal move retries and outcome.
            illegal_action_policy: Legacy fallback policy when illegal_policy is absent.
            chat_mode: Chat logging mode ("off", "ai-only", "all").
            chat_every_n: Record chat every N moves (>=1).
            ai_persona: Optional persona payload for AI players.
            rng_seed: Optional seed for random legal actions.
            run_id: Optional run identifier for replay output.
            sample_id: Optional sample identifier for replay output.
            replay_output_dir: Optional override for the replay output directory.
            replay_filename: Optional override for the replay filename.
            replay_live: Whether to update the replay file after each move.
            **_: Ignored extra keyword arguments.
        """

        _ = seat_mode
        self._game_type = str(game_type)
        self._core = core or make_core(self._game_type, config=core_config)
        resolved_ids = list(player_ids or self._default_player_ids(self._core))
        self._player_ids = [str(player_id) for player_id in resolved_ids]
        self._player_names = self._normalize_player_names(self._player_ids, player_names)
        self._player_id_map = {idx: player_id for idx, player_id in enumerate(self._player_ids)}
        self._player_index_map = {player_id: idx for idx, player_id in self._player_id_map.items()}
        self._formatter = formatter or StandardMahjongFormatter()
        self._parser = parser or StandardMahjongParser()
        self._renderer = renderer or StandardMahjongRenderer()
        self._chat_mode = str(chat_mode or "off")
        self._chat_every_n = max(1, int(chat_every_n))
        self._chat_queue = chat_queue
        self._chat_stop_event = Event()
        self._chat_thread: Optional[Thread] = None
        self._ai_persona = dict(ai_persona or {"enabled": False, "prompt": ""})
        self._rng = random.Random(rng_seed)
        self._rule_profile = "mahjong"
        self._run_id = str(run_id) if run_id else None
        self._sample_id = str(sample_id) if sample_id else None
        self._replay_output_dir = str(replay_output_dir) if replay_output_dir else None
        self._replay_filename = str(replay_filename) if replay_filename else None
        self._replay_live = bool(replay_live)
        action_id_to_text, action_text_to_id, action_id_to_raw = build_action_maps()
        self._action_id_to_text = action_id_to_text
        self._action_text_to_id = action_text_to_id
        self._action_id_to_raw = action_id_to_raw

        resolved_illegal_policy = dict(illegal_policy or {})
        if not resolved_illegal_policy and illegal_action_policy:
            fallback = str(illegal_action_policy or "").lower()
            # NOTE: Preserve legacy "reject" behavior by choosing a random legal move on failure.
            if fallback == "reject":
                fallback = "random"
            resolved_illegal_policy = {"retry": 0, "on_fail": fallback}
        if not resolved_illegal_policy:
            resolved_illegal_policy = {"retry": 0, "on_fail": "random"}
        self._illegal_policy = resolved_illegal_policy
        self._max_illegal = int(self._illegal_policy.get("retry", 0))
        self._illegal_on_fail = str(self._illegal_policy.get("on_fail", "loss"))

        self._move_log: list[MahjongMove] = []
        self._chat_log: list[MahjongChatMessage] = []
        self._hand_history: list[dict[int, Any] | None] = []
        self._final_result: Optional[GameResult] = None
        self._move_count = 0
        self._illegal_move_count = 0
        self._illegal_counts = {player_id: 0 for player_id in self._player_ids}
        self._last_move: Optional[str] = None
        self._replay_path: Optional[str] = None
        self._start_time_ms: Optional[int] = None

        # Initialize
        self.reset()

    def reset(self) -> None:
        """Reset the arena and core state."""

        # STEP 1: Reset the core and counters.
        self._core.reset()
        self._move_log = []
        self._chat_log = []
        self._hand_history = []
        self._move_count = 0
        self._illegal_move_count = 0
        self._illegal_counts = {player_id: 0 for player_id in self._player_ids}
        self._final_result = None
        self._last_move = None
        self._replay_path = None
        self._start_time_ms = int(time.time() * 1000)

        hands = self.get_all_hands()
        if hands is not None:
            self._hand_history.append(hands)
        if self._replay_live:
            self._save_replay(winner=None)
        self._start_chat_worker()

    def get_all_hands(self) -> Optional[dict[int, Any]]:
        """Return the current hands for all players, if available."""

        return self._core.get_all_hands()

    def get_active_player(self) -> str:
        """Return the active player identifier."""

        return self._player_id_map[self._core.get_active_player_id()]

    def observe(self, player_id: str) -> ArenaObservation:
        """Return an observation for the given player."""

        self._drain_chat_queue()

        # STEP 1: Resolve raw observation and legal actions.
        player_index = self._resolve_player_index(player_id)
        raw_obs = self._core.get_observation(player_index)
        legal_action_ids = self._core.get_legal_actions(player_index)

        # STEP 2: Format state and build the board text.
        public_state, private_state, legal_moves = self._formatter.format_observation(
            raw_obs,
            legal_action_ids,
        )
        chat_log = [] if self._chat_mode == "off" else list(self._chat_log)
        board_text = self._format_board_text(
            public_state,
            private_state,
            legal_moves,
            chat_log=chat_log if self._chat_mode != "off" else None,
        )

        # STEP 3: Build the observation payload.
        metadata = {
            "public_state": public_state,
            "private_state": private_state,
            "player_id": player_id,
            "player_ids": list(self._player_ids),
            "player_names": dict(self._player_names),
            "active_player_id": self.get_active_player(),
            "active_player": self.get_active_player(),
            "chat_log": chat_log,
            "chat_mode": self._chat_mode,
            "ai_persona": dict(self._ai_persona),
            "game_type": self._game_type,
            "last_move": self._last_move,
        }
        return ArenaObservation(
            board_text=board_text,
            legal_moves=legal_moves,
            active_player=self.get_active_player(),
            last_move=self._last_move,
            metadata=metadata,
        )

    def apply(self, action: MahjongAction | ArenaAction | str) -> Optional[GameResult]:
        """Apply an action and return the GameResult if the game ends."""

        if self._final_result is not None:
            return self._final_result

        self._drain_chat_queue()

        # STEP 1: Resolve the acting player and legal actions.
        player_id = self._resolve_action_player_id(action)
        if player_id != self.get_active_player():
            return self._handle_illegal(player_id, reason="wrong_player")

        player_index = self._resolve_player_index(player_id)
        legal_action_ids = self._core.get_legal_actions(player_index)

        # STEP 2: Parse the action payload.
        legal_moves = [self._formatter.format_action(action_id) for action_id in legal_action_ids]
        parsed = self._parser.parse(
            self._action_payload(action),
            legal_action_ids=legal_action_ids,
            legal_moves=legal_moves,
        )
        if parsed.error or parsed.action_id == -1:
            logger.warning("Invalid Mahjong action from {}: {}", player_id, parsed.error)
            return self._handle_illegal(
                player_id,
                reason=parsed.error or "invalid_action",
                legal_action_ids=legal_action_ids,
            )

        # STEP 3: Execute and check for terminal state.
        return self._execute_action(player_id, parsed, action=action)

    def is_terminal(self) -> bool:
        """Return True if the game has ended."""

        return self._final_result is not None or self._core.is_terminal()

    def build_result(self, *, result: Optional[str] = None, reason: Optional[str] = None) -> GameResult:
        """Build a final GameResult snapshot."""

        if self._final_result is not None:
            return self._final_result

        # STEP 1: Resolve payoffs and winner.
        payoffs = self._safe_payoffs()
        winner = self._resolve_winner(payoffs)
        resolved_result = result or ("win" if winner else "draw")
        if resolved_result != "loss" and winner is None:
            resolved_result = "draw"

        # STEP 2: Persist replay output and finalize result.
        final_board = self._snapshot_board()
        replay_path = self._save_replay(winner=winner)
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
        self._stop_chat_worker()
        return self._final_result

    def _execute_action(
        self,
        player_id: str,
        parsed: MahjongParsedAction,
        *,
        action: MahjongAction | ArenaAction | str,
    ) -> Optional[GameResult]:
        action_id = int(parsed.action_id)
        action_text = self._formatter.format_action(action_id)
        action_raw = self._action_id_to_raw.get(action_id, "")
        action_card = action_text if is_tile_action(action_raw) else None
        chat_text = self._resolve_chat_text(action, parsed)
        player_type = self._resolve_player_type(action)

        self._move_count += 1
        self._move_log.append(
            {
                "step": self._move_count,
                "player_id": player_id,
                "action_id": action_id,
                "action_text": action_text,
                "action_card": action_card,
                "chat": chat_text,
                "timestamp_ms": int(time.time() * 1000),
            }
        )
        self._last_move = action_text
        self._record_chat(player_id, chat_text, player_type)
        self._core.step(action_id)
        hands = self.get_all_hands()
        if hands is not None:
            self._hand_history.append(hands)
        if self._replay_live:
            self._save_replay(winner=None)

        if self._core.is_terminal():
            return self.build_result(result="win", reason="terminal")
        return None

    def _resolve_chat_text(
        self,
        action: MahjongAction | ArenaAction | str,
        parsed: MahjongParsedAction,
    ) -> Optional[str]:
        if parsed.chat_text:
            return str(parsed.chat_text)
        if isinstance(action, ArenaAction):
            chat_text = action.metadata.get("chat") if isinstance(action.metadata, dict) else None
            if chat_text:
                return str(chat_text)
        if isinstance(action, Mapping):
            chat_text = action.get("chat")
            if chat_text:
                return str(chat_text)
        return None

    def _resolve_player_type(self, action: MahjongAction | ArenaAction | str) -> Optional[str]:
        if isinstance(action, ArenaAction) and isinstance(action.metadata, dict):
            player_type = action.metadata.get("player_type")
            if player_type:
                return str(player_type)
        return None

    def _handle_illegal(
        self,
        player_id: str,
        *,
        reason: str,
        legal_action_ids: Optional[Sequence[int]] = None,
    ) -> Optional[GameResult]:
        if player_id in self._illegal_counts:
            self._illegal_counts[player_id] += 1
        self._illegal_move_count += 1
        if self._max_illegal < 0:
            return None
        if self._illegal_counts.get(player_id, 0) <= self._max_illegal:
            return None

        if self._illegal_on_fail == "random":
            fallback = self._resolve_random_action(legal_action_ids or [])
            if fallback is not None:
                parsed = MahjongParsedAction(
                    action_id=fallback,
                    action_text=self._formatter.format_action(fallback),
                )
                return self._execute_action(player_id, parsed, action="")

        if self._illegal_on_fail == "draw":
            winner = None
            result = "draw"
        else:
            winner = self._resolve_illegal_winner(player_id)
            result = "loss"

        replay_path = self._save_replay(winner=winner)
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
        self._stop_chat_worker()
        return self._final_result

    def _resolve_illegal_winner(self, offender_id: str) -> Optional[str]:
        for player_id in self._player_ids:
            if player_id != offender_id:
                return player_id
        return None

    def _resolve_random_action(self, legal_action_ids: Sequence[int]) -> Optional[int]:
        if not legal_action_ids:
            return None
        return int(self._rng.choice(list(legal_action_ids)))

    def _resolve_player_index(self, player_id: str) -> int:
        if player_id not in self._player_index_map:
            raise KeyError(f"Unknown player id: {player_id}")
        return int(self._player_index_map[player_id])

    def _default_player_ids(self, core: AbstractGameCore) -> list[str]:
        # Mahjong is strictly 4 players
        _ = core
        return [f"player_{idx}" for idx in range(4)]

    def _resolve_action_player_id(self, action: MahjongAction | ArenaAction | str) -> str:
        if isinstance(action, ArenaAction):
            return str(action.player)
        if isinstance(action, Mapping):
            player_id = action.get("player_id") or action.get("player")
            if player_id:
                return str(player_id)
        return self.get_active_player()

    def _action_payload(self, action: MahjongAction | ArenaAction | str) -> str | Mapping[str, Any]:
        if isinstance(action, ArenaAction):
            payload: dict[str, Any] = {"action": action.move, "raw": action.raw}
            if isinstance(action.metadata, dict):
                chat_text = action.metadata.get("chat")
                if chat_text:
                    payload["chat"] = chat_text
            return payload
        if isinstance(action, Mapping):
            return action
        return str(action)

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

    def _snapshot_board(self) -> str:
        try:
            active_player = self.get_active_player()
            observation = self.observe(active_player)
            return observation.board_text
        except Exception:
            return ""

    def _save_replay(self, *, winner: Optional[str]) -> Optional[str]:
        payload = self._build_replay_payload(winner=winner)
        output_path = self._resolve_replay_output_path()
        if output_path is None:
            return None
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
            self._replay_path = str(output_path)
            return self._replay_path
        except Exception as exc:
            logger.warning("Failed to write Mahjong replay to {}: {}", output_path, exc)
            return None

    def _build_replay_payload(self, *, winner: Optional[str]) -> dict[str, Any]:
        current_hands = self.get_all_hands() or {}
        active_player_id, active_player_idx, legal_moves = self._current_legal_snapshot()
        end_reason = self._resolve_end_reason()
        remaining_tiles = self._resolve_remaining_tiles()
        return {
            "player_ids": list(self._player_ids),
            "player_names": dict(self._player_names),
            "winner": winner,
            "result": self._final_result.result if self._final_result else None,
            "result_reason": self._final_result.reason if self._final_result else None,
            "end_reason": end_reason,
            "remaining_tiles": remaining_tiles,
            "moves": list(self._move_log),
            "current_hands": current_hands,
            "hand_history": list(self._hand_history),
            "chat_log": list(self._chat_log),
            "start_time_ms": self._start_time_ms,
            "active_player_id": active_player_id,
            "active_player_idx": active_player_idx,
            "legal_moves": list(legal_moves),
        }

    def _resolve_replay_output_path(self) -> Optional[Path]:
        if self._replay_output_dir:
            base_dir = Path(self._replay_output_dir)
        else:
            base_dir = Path(os.environ.get("GAGE_EVAL_SAVE_DIR", "./runs"))
        filename = self._replay_filename or "mahjong_replay.json"
        return base_dir / filename

    def _record_chat(self, player_id: str, chat_text: Optional[str], player_type: Optional[str]) -> None:
        if not chat_text or self._chat_mode == "off":
            return
        if self._chat_mode == "ai-only" and player_type == "human":
            return
        if self._chat_every_n > 1 and (self._move_count - 1) % self._chat_every_n != 0:
            return
        if player_type != "human":
            chat_text = self._format_chat_text(chat_text)
        self._chat_log.append(
            {
                "player_id": player_id,
                "text": chat_text,
                "timestamp_ms": int(time.time() * 1000),
            }
        )

    def _resolve_remaining_tiles(self) -> Optional[int]:
        try:
            core_env = getattr(self._core, "_env", None)
            game = getattr(core_env, "game", None)
            dealer = getattr(game, "dealer", None)
            deck = getattr(dealer, "deck", None)
            if isinstance(deck, (list, tuple)):
                return len(deck)
            cards = getattr(deck, "cards", None)
            if isinstance(cards, (list, tuple)):
                return len(cards)
        except Exception:
            return None
        return None

    def _format_chat_text(self, chat_text: str) -> str:
        text = str(chat_text)
        honor_map = {
            "east": "东",
            "south": "南",
            "west": "西",
            "north": "北",
            "red": "红中",
            "green": "发财",
            "white": "白板",
        }
        for key, value in honor_map.items():
            text = re.sub(rf"\\b{key}\\b", value, text, flags=re.IGNORECASE)
        def replace_tile(match: re.Match) -> str:
            suit = match.group(1).upper()
            rank = match.group(2)
            rank_map = {
                "1": "一",
                "2": "二",
                "3": "三",
                "4": "四",
                "5": "五",
                "6": "六",
                "7": "七",
                "8": "八",
                "9": "九",
            }
            suit_map = {"B": "条", "C": "万", "D": "筒"}
            return f"{rank_map.get(rank, rank)}{suit_map.get(suit, suit)}"
        text = re.sub(r"\\b([BCD])([1-9])\\b", replace_tile, text, flags=re.IGNORECASE)
        return text

    def _resolve_end_reason(self) -> Optional[str]:
        if self._final_result is None:
            return None
        if self._final_result.winner:
            return "hu"
        if self._final_result.result:
            return str(self._final_result.result)
        return None

    def _start_chat_worker(self) -> None:
        if self._chat_queue is None:
            return
        if self._chat_thread and self._chat_thread.is_alive():
            return
        self._chat_thread = Thread(target=self._chat_worker, daemon=True)
        self._chat_thread.start()

    def _stop_chat_worker(self) -> None:
        if not self._chat_thread:
            return
        self._chat_stop_event.set()
        self._chat_thread.join(timeout=1)

    def _chat_worker(self) -> None:
        while not self._chat_stop_event.is_set():
            had_updates = self._drain_chat_queue()
            if had_updates and self._replay_live:
                self._save_replay(winner=None)
            self._chat_stop_event.wait(0.2)

    def _drain_chat_queue(self) -> bool:
        if self._chat_queue is None:
            return False
        drained = False
        while True:
            try:
                payload = self._chat_queue.get_nowait()
            except Empty:
                break
            if not isinstance(payload, dict):
                continue
            chat_text = payload.get("text") or payload.get("chat") or payload.get("message")
            if not chat_text:
                continue
            player_id = payload.get("player_id") or payload.get("player") or payload.get("playerId")
            if not player_id:
                player_id = self.get_active_player()
            self._record_chat(str(player_id), str(chat_text), "human")
            drained = True
        return drained

    def _current_legal_snapshot(self) -> tuple[Optional[str], Optional[int], list[str]]:
        try:
            active_player_id = self.get_active_player()
            player_index = self._resolve_player_index(active_player_id)
            legal_action_ids = self._core.get_legal_actions(player_index)
            legal_moves = [self._formatter.format_action(action_id) for action_id in legal_action_ids]
            return active_player_id, player_index, legal_moves
        except Exception:
            return None, None, []

    @staticmethod
    def _format_board_text(
        public_state: dict[str, Any],
        private_state: dict[str, Any],
        legal_moves: Sequence[str],
        *,
        chat_log: Optional[Sequence[dict[str, str]]] = None,
    ) -> str:
        legal_preview = ", ".join(list(legal_moves)[:40]) if legal_moves else "none"
        lines: list[str] = [
            "Public State:",
            json.dumps(public_state, ensure_ascii=True),
            "",
            "Private State:",
            json.dumps(private_state, ensure_ascii=True),
            "",
            f"Legal Moves (preview): {legal_preview}",
        ]
        if chat_log is not None:
            lines.extend(
                [
                    "",
                    "Chat Log:",
                    json.dumps(list(chat_log), ensure_ascii=True),
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _normalize_player_names(
        player_ids: Sequence[str],
        player_names: Optional[dict[str, str]],
    ) -> dict[str, str]:
        names = dict(player_names or {})
        return {player_id: names.get(player_id, player_id) for player_id in player_ids}
