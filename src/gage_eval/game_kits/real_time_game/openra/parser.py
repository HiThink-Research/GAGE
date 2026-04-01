"""Minimal parser for OpenRA typed actions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Optional, Sequence

from gage_eval.registry import registry

DEFAULT_RETHINK_TEMPLATE = (
    "Your previous action could not be processed.\n"
    "Reason: {reason}.\n"
    "Your last output was: '{last_output}'.\n"
    "Instructions:\n"
    "- Output ONE JSON object on the last line.\n"
    '- Schema: {"action_id": "<legal_action_id>", "payload": {...}}.\n'
    "- If no payload is needed, use an empty object.\n"
    "Legal actions: {legal_moves}."
)


@dataclass(frozen=True)
class OpenRAParseResult:
    action_id: Optional[str]
    raw: str
    error: Optional[str]
    payload: dict[str, Any]


@registry.asset(
    "parser_impls",
    "openra_action_v1",
    desc="OpenRA typed action parser",
    tags=("openra", "parser", "gamekit"),
)
class OpenRAActionParser:
    """Parse minimal JSON or plain-text OpenRA action selections."""

    def parse(
        self,
        text: str,
        *,
        legal_moves: Optional[Iterable[str]] = None,
    ) -> OpenRAParseResult:
        raw = text or ""
        stripped = raw.strip()
        if not stripped:
            return OpenRAParseResult(None, raw, "empty_text", {})

        payload = self._extract_payload(stripped)
        action_id = (
            payload.get("action_id")
            or payload.get("actionId")
            or payload.get("action")
            or payload.get("move")
            if isinstance(payload, dict)
            else None
        )
        if action_id is None:
            action_id = stripped.splitlines()[-1].strip()
        if not action_id:
            return OpenRAParseResult(None, raw, "invalid_format", {})

        normalized_action_id = str(action_id).strip()
        if legal_moves is not None:
            legal_lookup = {
                str(item).strip().lower(): str(item)
                for item in legal_moves
                if str(item).strip()
            }
            resolved = legal_lookup.get(normalized_action_id.lower())
            if resolved is None:
                return OpenRAParseResult(normalized_action_id, raw, "illegal_move", {})
            normalized_action_id = resolved

        resolved_payload = payload.get("payload") if isinstance(payload, dict) else {}
        if not isinstance(resolved_payload, dict):
            resolved_payload = {}
        return OpenRAParseResult(normalized_action_id, raw, None, dict(resolved_payload))

    def build_rethink_prompt(
        self,
        *,
        last_output: str,
        reason: str,
        legal_moves: Sequence[str],
    ) -> str:
        return DEFAULT_RETHINK_TEMPLATE.format(
            reason=reason,
            last_output=last_output,
            legal_moves=", ".join(legal_moves),
        )

    @staticmethod
    def build_action_dict(*, player: str, parse_result: OpenRAParseResult) -> dict[str, Any]:
        return {
            "player": str(player),
            "move": str(parse_result.action_id or ""),
            "raw": json.dumps(
                {
                    "action_id": parse_result.action_id,
                    "payload": dict(parse_result.payload),
                },
                ensure_ascii=False,
            ),
        }

    @staticmethod
    def _extract_payload(text: str) -> dict[str, Any] | None:
        if not (text.startswith("{") and text.endswith("}")):
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
        return None


__all__ = ["OpenRAActionParser", "OpenRAParseResult"]
