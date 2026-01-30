"""Mahjong-specific move parser."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from gage_eval.registry import registry
from gage_eval.role.arena.games.mahjong.mapping import build_action_maps
from gage_eval.role.arena.games.mahjong.parsers.base import MahjongMoveParser, MahjongParsedAction

DEFAULT_RETHINK_TEMPLATE = (
    "Your previous action could not be processed.\n"
    "Reason: {reason}.\n"
    "Your last output was: '{last_output}'.\n"
    "Instructions:\n"
    "- Output exactly one legal action string from the list below.\n"
    '- If you include chat, use JSON: {"action": "<action>", "chat": "<short line>"}.\n'
    "Legal moves: {legal_moves}."
)

@registry.asset("parser_impls", "mahjong_v1", desc="Mahjong move parser")
class StandardMahjongParser(MahjongMoveParser):
    """Parses text or JSON payloads into Mahjong action ids."""

    def __init__(self, **kwargs: Any) -> None:
        action_id_to_text, action_text_to_id, _ = build_action_maps()
        self._action_id_to_text = action_id_to_text
        self._action_text_to_id = action_text_to_id

    def parse(
        self,
        payload: str | Mapping[str, Any],
        *,
        legal_action_ids: Optional[Sequence[int]] = None,
        legal_moves: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> MahjongParsedAction:
        """Parse a Mahjong action payload and optionally validate against legal moves.

        Args:
            payload: Raw action payload (text or JSON-like mapping).
            legal_action_ids: Optional legal action ids to validate against.
            legal_moves: Optional legal action strings to validate against.

        Returns:
            Parsed action payload with optional error info.
        """

        # STEP 1: Normalize payload into action fields.
        action_id: Optional[int] = None
        action_text: Optional[str] = None
        chat_text: Optional[str] = None
        raw_text: Optional[str] = None

        if isinstance(payload, Mapping):
            action_id = payload.get("action_id")
            action_text = payload.get("action")
            chat_text = payload.get("chat")
            raw_text = payload.get("raw")
        else:
            raw_text = str(payload)
            parsed_payload = self._try_parse_json(raw_text)
            if isinstance(parsed_payload, Mapping):
                action_id = parsed_payload.get("action_id")
                action_text = parsed_payload.get("action")
                chat_text = parsed_payload.get("chat")
                raw_text = parsed_payload.get("raw", raw_text)
            else:
                action_text, chat_text = self._split_action_and_chat(raw_text)

        # STEP 2: Resolve action text and action id.
        error_msg = None
        normalized_text = self._normalize_action_text(action_text or "")
        resolved_id = self._resolve_action_id(action_id, normalized_text)

        if resolved_id is None:
            error_msg = f"Unknown action: {normalized_text}"
            resolved_id = -1
        resolved_text = self._action_id_to_text.get(resolved_id, normalized_text)

        # STEP 3: Validate against legal actions when provided.
        if resolved_id != -1:
            if legal_action_ids is not None and resolved_id not in legal_action_ids:
                error_msg = f"Illegal action id {resolved_id} for text '{resolved_text}'"
            elif legal_moves is not None and not self._is_legal_text(resolved_text, legal_moves):
                error_msg = f"Illegal action text '{resolved_text}'"

        return MahjongParsedAction(
            action_id=resolved_id,
            action_text=resolved_text,
            chat_text=chat_text,
            raw=raw_text,
            error=error_msg,
        )

    def build_rethink_prompt(
        self,
        *,
        last_output: str,
        reason: str,
        legal_moves: Sequence[str],
    ) -> str:
        """Build a retry prompt when an illegal move is detected.

        Args:
            last_output: The previous model output.
            reason: Explanation for why the move is invalid.
            legal_moves: List of legal action strings.

        Returns:
            A formatted prompt for rethinking.
        """

        legal_block = ", ".join(list(legal_moves))
        return DEFAULT_RETHINK_TEMPLATE.format(
            reason=reason,
            last_output=last_output,
            legal_moves=legal_block,
        )

    def _resolve_action_id(self, action_id: Optional[Any], action_text: str) -> Optional[int]:
        if action_id is not None:
            try:
                return int(action_id)
            except (TypeError, ValueError):
                return None
        if not action_text:
            return None
        lowered = action_text.lower()
        if lowered in self._action_text_to_id:
            return int(self._action_text_to_id[lowered])
        if action_text.isdigit():
            candidate = int(action_text)
            if candidate in self._action_id_to_text:
                return candidate
        return None

    def _normalize_action_text(self, action_text: str) -> str:
        normalized = str(action_text or "").strip()
        if not normalized:
            return ""
        lowered = normalized.lower()
        for prefix in ("play:", "out:", "action:", "move:"):
            if lowered.startswith(prefix):
                return normalized[len(prefix) :].strip()
        return normalized

    def _is_legal_text(self, action_text: str, legal_moves: Sequence[str]) -> bool:
        lowered = action_text.lower()
        return lowered in {str(move).lower() for move in legal_moves}

    def _split_action_and_chat(self, raw_text: str) -> tuple[str, Optional[str]]:
        stripped = str(raw_text or "").strip()
        if not stripped:
            return "", None
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not lines:
            return "", None
        action_text = self._normalize_action_text(lines[0])
        if len(lines) == 1:
            return action_text, None
        chat_text = " ".join(lines[1:]).strip()
        return action_text, self._strip_chat_prefix(chat_text)

    def _strip_chat_prefix(self, chat_text: str) -> str:
        lowered = chat_text.lower()
        for prefix in ("chat:", "say:", "table talk:", "talk:"):
            if lowered.startswith(prefix):
                return chat_text[len(prefix) :].strip()
        return chat_text

    def _try_parse_json(self, raw_text: str) -> Optional[Mapping[str, Any]]:
        stripped = raw_text.strip()
        if not stripped.startswith("{"):
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, Mapping):
            return parsed
        return None
