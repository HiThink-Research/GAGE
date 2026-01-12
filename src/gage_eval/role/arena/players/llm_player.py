"""LLM-backed arena player implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from loguru import logger

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.arena.interfaces import MoveParser
from gage_eval.role.arena.types import ArenaAction, ArenaObservation
from gage_eval.utils.messages import stringify_message_content


class LLMPlayer:
    """Arena player that delegates decisions to a DUT model adapter."""

    def __init__(
        self,
        *,
        name: str,
        adapter_id: str,
        role_manager,
        sample: Dict[str, Any],
        parser: MoveParser,
        trace: Optional[ObservabilityTrace] = None,
        max_retries: int = 0,
        legal_moves_limit: int = 40,
        sampling_params: Optional[Dict[str, Any]] = None,
        fallback_policy: str = "none",
    ) -> None:
        self.name = name
        self._adapter_id = adapter_id
        self._role_manager = role_manager
        self._sample = sample
        self._parser = parser
        self._trace = trace
        self._max_retries = max(0, int(max_retries))
        self._legal_moves_limit = max(0, int(legal_moves_limit))
        self._sampling_params = dict(sampling_params or {})
        self._fallback_policy = str(fallback_policy or "none").lower()
        self._base_messages = list(sample.get("messages") or [])

    def think(self, observation: ArenaObservation) -> ArenaAction:
        """Produce an action using the LLM adapter."""

        prompt_text = self._format_observation(observation)
        messages = self._base_messages + [self._build_user_message(prompt_text)]

        # STEP 1: Run the primary request.
        raw_text = self._invoke_model(messages)
        parse_result = self._parser.parse(raw_text, legal_moves=observation.legal_moves)

        # STEP 2: Retry with rethink prompts when parsing fails or is illegal.
        retries = 0
        while parse_result.error and retries < self._max_retries:
            rethink_prompt = self._parser.build_rethink_prompt(
                last_output=raw_text,
                reason=parse_result.error,
                legal_moves=self._truncate_legal_moves(observation.legal_moves),
            )
            raw_text = self._invoke_model(self._base_messages + [self._build_user_message(rethink_prompt)])
            parse_result = self._parser.parse(raw_text, legal_moves=observation.legal_moves)
            retries += 1

        if parse_result.error and observation.legal_moves and self._fallback_policy == "first_legal":
            fallback_move = observation.legal_moves[0]
            logger.warning(
                "LLMPlayer {} fallback to legal move {} due to {}",
                self.name,
                fallback_move,
                parse_result.error,
            )
            metadata = self._build_action_metadata(parse_result)
            metadata["error"] = parse_result.error
            metadata["fallback"] = "first_legal"
            return ArenaAction(
                player=self.name,
                move=fallback_move,
                raw=raw_text,
                metadata=metadata,
            )

        metadata = self._build_action_metadata(parse_result)
        if parse_result.error:
            metadata["error"] = parse_result.error
        return ArenaAction(
            player=self.name,
            move=parse_result.coord or "",
            raw=raw_text,
            metadata=metadata,
        )

    def _invoke_model(self, messages: Sequence[Dict[str, Any]]) -> str:
        payload = {
            "sample": self._sample,
            "messages": messages,
            "sampling_params": self._sampling_params,
            "usage": "arena_move",
        }
        if self._trace:
            payload["trace"] = self._trace
        with self._role_manager.borrow_role(self._adapter_id) as role:
            output = role.invoke(payload, self._trace) if role else {}
        raw_text = _extract_text(output)
        logger.debug("LLMPlayer {} output={}", self.name, raw_text)
        return raw_text

    def _format_observation(self, observation: ArenaObservation) -> str:
        if self._should_use_card_prompt(observation):
            return self._format_card_observation(observation)
        return self._format_grid_observation(observation)

    def _format_grid_observation(self, observation: ArenaObservation) -> str:
        legal_moves = self._truncate_legal_moves(observation.legal_moves)
        legal_hint = ", ".join(legal_moves) if legal_moves else "none"
        active_player = _format_player_label(observation, observation.active_player)

        lines = [
            f"Active player: {active_player}",
            f"Opponent last move: {observation.last_move or 'First move'}",
            "\nCurrent Board:",
            observation.board_text,
            "\nStatus:",
            f"- Legal moves (truncated): {legal_hint}",
            "\nInstructions:",
            "- Analyze the board.",
            "- Select the best coordinate for your move.",
            "- Output your move as a single coordinate (e.g., 'H8').",
        ]
        return "\n".join(lines)

    def _format_card_observation(self, observation: ArenaObservation) -> str:
        legal_moves = self._truncate_legal_moves(observation.legal_moves)
        legal_hint = ", ".join(legal_moves) if legal_moves else "none"
        active_player = _format_player_label(observation, observation.active_player)
        chat_mode = str(observation.metadata.get("chat_mode", "off")).lower()
        include_chat = chat_mode in {"ai-only", "all"}
        instructions = [
            "- Choose exactly one legal action string from the legal moves.",
        ]
        if include_chat:
            instructions.extend(
                [
                    "- Include a short table-talk line every turn.",
                    '- Output JSON: {"action": "<action>", "chat": "<short line>"}',
                ]
            )
        else:
            instructions.append("- Output the action string only.")
        lines = [
            f"Active player: {active_player}",
            f"Opponent last move: {observation.last_move or 'First move'}",
            "\nCurrent State:",
            observation.board_text,
            "\nStatus:",
            f"- Legal moves (preview): {legal_hint}",
            "\nInstructions:",
            *instructions,
        ]
        return "\n".join(lines)

    def _should_use_card_prompt(self, observation: ArenaObservation) -> bool:
        metadata = self._sample.get("metadata") if isinstance(self._sample, dict) else {}
        game_type = str(metadata.get("game_type", "")).lower()
        if game_type == "doudizhu":
            return True
        if isinstance(observation.metadata.get("public_state"), dict):
            return True
        return "Public State:" in observation.board_text

    def _truncate_legal_moves(self, legal_moves: Sequence[str]) -> Sequence[str]:
        if self._legal_moves_limit <= 0:
            return []
        if len(legal_moves) <= self._legal_moves_limit:
            return list(legal_moves)
        return list(legal_moves[: self._legal_moves_limit])

    @staticmethod
    def _build_user_message(text: str) -> Dict[str, Any]:
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    def _build_action_metadata(self, parse_result) -> Dict[str, Any]:
        metadata = {"player_type": "backend"}
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
