"""Human-backed arena player implementation."""

from __future__ import annotations

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
        parse_result = self._parser.parse(raw_text, legal_moves=observation.legal_moves)
        if parse_result.error:
            logger.warning("HumanPlayer {} provided illegal move: {}", self.name, parse_result.error)
        return ArenaAction(
            player=self.name,
            move=parse_result.coord or "",
            raw=raw_text,
            metadata={"error": parse_result.error} if parse_result.error else {},
        )

    def _format_observation(self, observation: ArenaObservation) -> str:
        active_player = _format_player_label(observation, observation.active_player)
        lines = [
            f"Active player: {active_player}",
            f"Opponent last move: {observation.last_move or 'First move'}",
            "\nBoard:",
            observation.board_text,
            "\nInstructions:",
            "- Enter a single coordinate (e.g. 'H8').",
            "- Win by 5 in a row. Overlines allowed.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_user_message(text: str) -> Dict[str, Any]:
        return {"role": "user", "content": [{"type": "text", "text": text}]}


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
