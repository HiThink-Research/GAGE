"""Human-backed arena player implementation."""

from __future__ import annotations

import threading
import time
from queue import Queue, Empty
from typing import Any, Dict, Optional, Sequence

from loguru import logger

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.arena.interfaces import MoveParser
from gage_eval.role.arena.types import ArenaAction, ArenaObservation
from gage_eval.utils.messages import stringify_message_content


class HumanPlayer:
    """Arena player that delegates decisions to a HumanAdapter."""

    def __init__(
        self,
        *,
        name: str,
        adapter_id: str,
        role_manager,
        sample: Dict[str, Any],
        parser: MoveParser,
        trace: Optional[ObservabilityTrace] = None,
        action_queue=None,
    ) -> None:
        self.name = name
        self._adapter_id = adapter_id
        self._role_manager = role_manager
        self._sample = sample
        self._parser = parser
        self._trace = trace
        self._action_queue = action_queue
        self._decision_queue: "Queue[ArenaAction]" = Queue()
        self._think_thread: Optional[threading.Thread] = None
        self._think_lock = threading.Lock()
        self._pending_error: Optional[BaseException] = None

    def think(self, observation: ArenaObservation) -> ArenaAction:
        """Return a human-provided action."""

        prompt_text = self._format_observation(observation)
        payload = {
            "sample": self._sample,
            "prompt": prompt_text,
            "messages": [self._build_user_message(prompt_text)],
        }
        if self._action_queue is not None:
            payload["action_queue"] = self._action_queue
        if self._trace:
            payload["trace"] = self._trace
        with self._role_manager.borrow_role(self._adapter_id) as role:
            output = role.invoke(payload, self._trace) if role else {}
        raw_text = _extract_text(output)
        parse_result = self._parser.parse(raw_text, legal_moves=observation.legal_actions_items)
        if parse_result.error:
            logger.warning("HumanPlayer {} provided illegal move: {}", self.name, parse_result.error)
        metadata = self._build_action_metadata(parse_result)
        if parse_result.error:
            metadata["error"] = parse_result.error
        return ArenaAction(
            player=self.name,
            move=parse_result.coord or "",
            raw=raw_text,
            metadata=metadata,
        )

    def start_thinking(self, observation: ArenaObservation, *, deadline_ms: int = 100) -> None:
        """Start waiting for the next human action on a background thread."""

        del deadline_ms
        with self._think_lock:
            if self._think_thread and self._think_thread.is_alive():
                return
            self._pending_error = None

            # STEP 1: Block on human input in a background thread to keep schedulers responsive.
            def _worker(obs: ArenaObservation = observation) -> None:
                start_s = time.perf_counter()
                try:
                    action = self.think(obs)
                except BaseException as exc:  # pragma: no cover - defensive
                    self._pending_error = exc
                    return
                latency_ms = max(0, int((time.perf_counter() - start_s) * 1000))
                action_meta = dict(action.metadata or {})
                action_meta.setdefault("latency_ms", latency_ms)
                self._decision_queue.put(
                    ArenaAction(
                        player=action.player,
                        move=action.move,
                        raw=action.raw,
                        metadata=action_meta,
                        hold_ticks=action.hold_ticks,
                    )
                )

            self._think_thread = threading.Thread(target=_worker, daemon=True)
            self._think_thread.start()

    def has_action(self) -> bool:
        """Return True if a completed action is available."""

        return not self._decision_queue.empty() or self._pending_error is not None

    def pop_action(self) -> ArenaAction:
        """Return the next available action.

        Raises:
            RuntimeError: If the background worker failed to produce an action.
        """

        if self._pending_error is not None:
            error = self._pending_error
            self._pending_error = None
            raise RuntimeError(f"HumanPlayer '{self.name}' failed to produce an action: {error}") from error
        try:
            return self._decision_queue.get_nowait()
        except Empty as exc:
            raise RuntimeError(f"HumanPlayer '{self.name}' pop_action called without available action") from exc

    def _format_observation(self, observation: ArenaObservation) -> str:
        active_player = _format_player_label(observation, observation.active_player)
        lines = [
            f"Active player: {active_player}",
            f"Opponent last move: {observation.last_action or 'First move'}",
            "\nBoard:",
            observation.view_text,
            "\nInstructions:",
            "- Enter a single coordinate (e.g. 'H8').",
            "- Win by 5 in a row. Overlines allowed.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_user_message(text: str) -> Dict[str, Any]:
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    def _build_action_metadata(self, parse_result) -> Dict[str, Any]:
        metadata = {"player_type": "human"}
        chat_text = getattr(parse_result, "chat_text", None)
        if chat_text:
            metadata["chat"] = str(chat_text)
        return metadata


def _format_player_label(observation: ArenaObservation, player_id: str) -> str:
    names = observation.metadata.get("player_names")
    if isinstance(names, dict):
        display_name = names.get(player_id)
        if display_name and display_name != player_id:
            return f"{display_name} ({player_id})"
    return player_id


def _extract_text(output: Any) -> str:
    if isinstance(output, dict):
        if isinstance(output.get("answer"), str):
            return output["answer"]
        if isinstance(output.get("text"), str):
            return output["text"]
        message = output.get("message")
        if isinstance(message, dict):
            return stringify_message_content(message.get("content"))
        messages = output.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                return stringify_message_content(last.get("content"))
    return "" if output is None else str(output)
