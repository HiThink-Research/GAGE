"""Human-backed arena player implementation."""

from __future__ import annotations

import json
from queue import Empty, Queue
from threading import Lock
import time
from typing import Any, Dict, Optional, Sequence

from loguru import logger

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.arena.action_trace import attach_trace_action_applied
from gage_eval.role.arena.human_input_protocol import (
    action_matches_route,
    dump_action_payload,
    extract_action_text,
    parse_action_payload,
)
from gage_eval.role.arena.interfaces import MoveParser
from gage_eval.role.arena.types import ArenaAction, ArenaObservation
from gage_eval.utils.messages import stringify_message_content
from gage_eval.utils.samples import extract_sample_id


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
        timeout_ms: Optional[int] = None,
        timeout_fallback_move: Optional[str] = None,
    ) -> None:
        self.name = name
        self._adapter_id = adapter_id
        self._role_manager = role_manager
        self._sample = sample
        self._sample_id = extract_sample_id(sample)
        self._parser = parser
        self._trace = trace
        self._action_queue = action_queue
        self._default_timeout_ms = None if timeout_ms is None else max(1, int(timeout_ms))
        self._timeout_fallback_move = (
            None if timeout_fallback_move is None else str(timeout_fallback_move)
        )
        self._timeout_ms: Optional[int] = None
        self._async_lock = Lock()
        self._async_queue: Queue[ArenaAction] = Queue()
        self._async_inflight = False
        self._async_start_ts: Optional[float] = None
        self._pending_observation: Optional[ArenaObservation] = None

    def think(self, observation: ArenaObservation) -> ArenaAction:
        """Return a human-provided action."""

        prompt_text = self._format_observation(observation)
        payload = {
            "sample": self._sample,
            "sample_id": self._sample_id,
            "player_id": self.name,
            "prompt": prompt_text,
            "messages": [self._build_user_message(prompt_text)],
        }
        if self._timeout_ms is not None:
            payload["timeout_ms"] = self._timeout_ms
        if _is_vizdoom_observation(observation):
            payload["default_action"] = "0"
        if self._action_queue is not None:
            payload["action_queue"] = self._action_queue
        if self._trace:
            payload["trace"] = self._trace
        deferred_actions: list[Any] = []
        raw_payload: Any = None
        try:
            while True:
                if raw_payload is None:
                    with self._role_manager.borrow_role(self._adapter_id) as role:
                        output = role.invoke(payload, self._trace) if role else {}
                    raw_payload = _extract_text(output)

                action_payload = parse_action_payload(raw_payload)
                if not self._matches_action_payload(action_payload):
                    self._log_route_mismatch(action_payload, source="sync")
                    if self._action_queue is not None:
                        deferred_actions.append(raw_payload)
                        raw_payload = self._dequeue_action_blocking()
                        continue
                    raw_payload = None
                    continue

                action = self._parse_human_action(observation, action_payload)
                if action is not None:
                    return action
                raw_payload = None
        finally:
            for queued_action in deferred_actions:
                self._requeue_action(queued_action)

    def start_thinking(self, observation: ArenaObservation, *, deadline_ms: Optional[int] = None) -> bool:
        """Start thinking asynchronously if no request is in-flight."""

        self._timeout_ms = deadline_ms if deadline_ms is not None else self._default_timeout_ms
        with self._async_lock:
            if self._async_inflight:
                return False
            self._async_inflight = True
            self._async_start_ts = time.monotonic()
            self._pending_observation = observation

        adapter = getattr(self._role_manager, "get_adapter", lambda _id: None)(self._adapter_id)
        ensure_input_ready = getattr(adapter, "ensure_input_ready", None)
        if callable(ensure_input_ready):
            ensure_input_ready()

        return True

    def has_action(self) -> bool:
        """Return True if an async action is ready."""

        if not self._async_queue.empty():
            return True
        with self._async_lock:
            if not self._async_inflight:
                return False
            observation = self._pending_observation
            started = self._async_start_ts
        if observation is None or started is None:
            return False
        adapter = getattr(self._role_manager, "get_adapter", lambda _id: None)(self._adapter_id)
        poll_action = getattr(adapter, "poll_action", None)
        if callable(poll_action):
            action_text = poll_action(timeout_ms=0, default_action=None)
            if action_text:
                action_payload = parse_action_payload(action_text)
                if not self._matches_action_payload(action_payload):
                    self._log_route_mismatch(action_payload, source="async_adapter")
                    self._requeue_action(dump_action_payload(action_payload))
                    return False
                action = self._parse_human_action(observation, action_payload)
                if action is not None:
                    self._async_queue.put(action)
                    return True
        if self._action_queue is not None:
            get_nowait = getattr(self._action_queue, "get_nowait", None)
            if callable(get_nowait):
                try:
                    queued_action = get_nowait()
                except Empty:
                    queued_action = None
                if queued_action is not None:
                    action_payload = parse_action_payload(queued_action)
                    if not self._matches_action_payload(action_payload):
                        self._log_route_mismatch(action_payload, source="async_queue")
                        self._requeue_action(dump_action_payload(action_payload))
                        return False
                    if extract_action_text(action_payload) is None:
                        return False
                    action = self._parse_human_action(observation, action_payload)
                    if action is not None:
                        self._async_queue.put(action)
                        return True
        if self._timeout_ms is not None:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            if elapsed_ms >= float(self._timeout_ms):
                fallback_move = self._resolve_timeout_fallback_move(observation)
                if fallback_move is None:
                    return False
                timeout_action = ArenaAction(
                    player=self.name,
                    move=fallback_move,
                    raw=str(fallback_move),
                    metadata={"player_type": "human", "fallback": "timeout_noop"},
                )
                self._async_queue.put(timeout_action)
                return True
        return False

    def pop_action(self) -> ArenaAction:
        """Pop the next async action."""

        action = self._async_queue.get_nowait()
        with self._async_lock:
            self._async_inflight = False
            self._pending_observation = None
            self._async_start_ts = None
        return action

    def wait_for_pending(self, timeout_s: float = 1.0) -> None:
        """Wait briefly for any in-flight async call to finish."""

        _ = timeout_s
        return None

    def _format_observation(self, observation: ArenaObservation) -> str:
        if _is_vizdoom_observation(observation):
            legal_moves = observation.legal_actions_items
            legal_hint = ", ".join(legal_moves) if legal_moves else "none"
            lines = [
                "You are playing ViZDoom.",
                observation.view_text,
                "",
                "Instructions:",
                f"- Legal actions: {legal_hint}.",
                "- Enter ONE action id as an integer (e.g., 2).",
            ]
            return "\n".join(lines)
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

    def _resolve_timeout_fallback_move(self, observation: ArenaObservation) -> Optional[str]:
        if self._timeout_fallback_move is not None:
            return self._timeout_fallback_move
        legal_moves = observation.legal_actions_items
        if legal_moves:
            return str(legal_moves[0])
        if _is_vizdoom_observation(observation):
            return "0"
        return "0"

    @staticmethod
    def _build_user_message(text: str) -> Dict[str, Any]:
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    def _build_action_metadata(self, parse_result) -> Dict[str, Any]:
        metadata = {"player_type": "human", "retry_count": 0}
        chat_text = getattr(parse_result, "chat_text", None)
        if chat_text:
            metadata["chat"] = str(chat_text)
        hold_ticks = getattr(parse_result, "hold_ticks", None)
        if hold_ticks is not None:
            try:
                metadata["hold_ticks"] = max(1, int(hold_ticks))
            except (TypeError, ValueError):
                pass
        return metadata

    def _requeue_action(self, queued_action: Any) -> None:
        queue = self._action_queue
        if queue is None:
            return
        try:
            put_nowait = getattr(queue, "put_nowait", None)
            if callable(put_nowait):
                put_nowait(queued_action)
                return
            put = getattr(queue, "put", None)
            if callable(put):
                put(queued_action)
        except Exception:
            return

    def _dequeue_action_blocking(self) -> Any:
        queue = self._action_queue
        if queue is None:
            return None
        get = getattr(queue, "get", None)
        if not callable(get):
            return None
        return get()

    def _log_route_mismatch(self, payload: Dict[str, Any], *, source: str) -> None:
        target_player_id = payload.get("player_id")
        target_sample_id = payload.get("sample_id")
        logger.debug(
            "HumanPlayer {} ignored input for player={} sample={} from {} while waiting for sample={}.",
            self.name,
            target_player_id,
            target_sample_id,
            source,
            self._sample_id,
        )

    def _parse_human_action(
        self,
        observation: ArenaObservation,
        action_payload: Any,
    ) -> Optional[ArenaAction]:
        payload = parse_action_payload(action_payload)
        if not self._matches_action_payload(payload):
            self._log_route_mismatch(payload, source="parse")
            return None
        parse_text = extract_action_text(payload)
        if parse_text is None:
            return None
        raw_text = str(payload.get("raw") or parse_text)
        parse_result = self._parser.parse(parse_text, legal_moves=observation.legal_actions_items)
        if parse_result.error:
            logger.warning("HumanPlayer {} provided illegal move: {}", self.name, parse_result.error)
        metadata = self._build_action_metadata(parse_result)
        metadata = attach_trace_action_applied(
            metadata,
            observation=observation,
            move=parse_result.coord or "",
        )
        if parse_result.error:
            metadata["error"] = parse_result.error
        return ArenaAction(
            player=self.name,
            move=parse_result.coord or "",
            raw=raw_text,
            metadata=metadata,
        )

    def _matches_action_payload(self, payload: Dict[str, Any]) -> bool:
        return action_matches_route(
            payload,
            sample_id=self._sample_id,
            player_id=self.name,
        )


def _format_player_label(observation: ArenaObservation, player_id: str) -> str:
    names = observation.metadata.get("player_names")
    if isinstance(names, dict):
        display_name = names.get(player_id)
        if display_name and display_name != player_id:
            return f"{display_name} ({player_id})"
    return player_id


def _is_vizdoom_observation(observation: ArenaObservation) -> bool:
    metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
    game_type = str(metadata.get("game_type", "")).lower()
    return game_type == "vizdoom"


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
