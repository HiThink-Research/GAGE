"""Doudizhu-specific move parser."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional, Sequence

from gage_eval.registry import registry
from gage_eval.role.arena.games.doudizhu.parsers.base import CardMoveParser, ParsedAction


@registry.asset(
    "parser_impls",
    "doudizhu_v1",
    desc="Doudizhu move parser (RLCard action strings)",
    tags=("doudizhu", "parser", "card"),
)
class DoudizhuMoveParser(CardMoveParser):
    """Parses text or JSON payloads into Doudizhu action ids."""

    def __init__(self, *, action_text_to_id: Optional[Mapping[str, int]] = None) -> None:
        """Initialize the parser.

        Args:
            action_text_to_id: Optional mapping for action text encoding.
        """

        self._action_text_to_id = dict(action_text_to_id or self._load_action_map())
        self._pass_aliases = {"pass", "skip", "fold", "no"}
        self._pair_markers = {"pair", "double", "dui", "\u5bf9"}

    def parse(
        self,
        payload: str | Mapping[str, Any],
        *,
        legal_action_ids: Optional[Sequence[int]] = None,
    ) -> ParsedAction:
        """Parse an action payload into a normalized Doudizhu action.

        Args:
            payload: Raw action payload (text or JSON-like mapping).
            legal_action_ids: Optional legal action ids to validate against.

        Returns:
            Parsed action payload.
        """

        # STEP 1: Normalize the payload into action fields.
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
        if action_id is None:
            resolved_text = self._resolve_action_text(action_text or "")
            action_text = resolved_text
            action_id = self._encode_action(resolved_text)
        else:
            action_id = int(action_id)
            if action_text is None:
                action_text = str(action_id)

        # STEP 3: Validate against legal actions when provided.
        if legal_action_ids is not None and action_id not in legal_action_ids:
            raise ValueError(f"Illegal action id: {action_id}")

        return ParsedAction(
            action_id=action_id,
            action_text=str(action_text),
            chat_text=chat_text,
            raw=raw_text,
        )

    def _resolve_action_text(self, action_text: str) -> str:
        normalized = str(action_text or "").strip()
        if not normalized:
            raise ValueError("Empty action text")
        lowered = normalized.lower()
        if lowered in self._pass_aliases:
            return "pass"
        if normalized in self._action_text_to_id:
            return normalized
        candidate = self._extract_cards(normalized)
        if self._contains_pair_marker(normalized) and len(candidate) == 1:
            candidate = candidate * 2
        if candidate and candidate in self._action_text_to_id:
            return candidate
        raise ValueError(f"Unknown action text: {action_text}")

    def _encode_action(self, action_text: str) -> int:
        if action_text not in self._action_text_to_id:
            raise ValueError(f"Unknown action text: {action_text}")
        return int(self._action_text_to_id[action_text])

    def _split_action_and_chat(self, raw_text: str) -> tuple[str, Optional[str]]:
        stripped = str(raw_text or "").strip()
        if not stripped:
            return "", None
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not lines:
            return "", None
        action_text = self._strip_action_prefix(lines[0])
        if len(lines) == 1:
            return action_text, None
        chat_text = " ".join(lines[1:]).strip()
        return action_text, self._strip_chat_prefix(chat_text)

    def _strip_action_prefix(self, action_text: str) -> str:
        lowered = action_text.lower()
        for prefix in ("action:", "move:", "play:"):
            if lowered.startswith(prefix):
                return action_text[len(prefix):].strip()
        return action_text

    def _strip_chat_prefix(self, chat_text: str) -> str:
        lowered = chat_text.lower()
        for prefix in ("chat:", "say:", "table talk:", "talk:"):
            if lowered.startswith(prefix):
                return chat_text[len(prefix):].strip()
        return chat_text

    def _extract_cards(self, text: str) -> str:
        tokens = re.findall(r"[3456789TJQKA2BR]", text.upper())
        return "".join(tokens)

    def _contains_pair_marker(self, text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in self._pair_markers) or "\u5bf9" in text

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

    def _load_action_map(self) -> dict[str, int]:
        try:
            from rlcard.games.doudizhu import utils as doudizhu_utils
        except Exception as exc:
            raise RuntimeError("rlcard is required to parse doudizhu actions") from exc

        return {str(key): int(value) for key, value in doudizhu_utils.ACTION_2_ID.items()}
