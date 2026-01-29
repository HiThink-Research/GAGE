"""Parsing utilities for stable-retro action JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from gage_eval.registry import registry

DEFAULT_RETHINK_TEMPLATE = (
    "Your previous action could not be processed.\n"
    "Reason: {reason}.\n"
    "Your last output was: '{last_output}'.\n"
    "Instructions:\n"
    "- Output ONE JSON object on the last line.\n"
    '- Schema: {"move": "<legal_move>", "hold_ticks": <int>}.\n'
    "Legal moves: {legal_moves}."
)


@dataclass(frozen=True)
class RetroParseResult:
    """Represents the parsed retro action and any parsing error."""

    move: Optional[str]
    coord: Optional[str]
    raw: str
    error: Optional[str]
    hold_ticks: Optional[int] = None


@registry.asset(
    "parser_impls",
    "retro_action_v1",
    desc="Stable-retro macro action parser (JSON + hold_ticks)",
    tags=("retro", "parser", "arena"),
)
class RetroActionParser:
    """Parse macro actions for stable-retro environments."""

    def __init__(
        self,
        *,
        hold_ticks_min: int = 1,
        hold_ticks_max: int = 20,
        default_hold_ticks: int = 6,
    ) -> None:
        """Initialize the parser.

        Args:
            hold_ticks_min: Minimum allowed hold_ticks value.
            hold_ticks_max: Maximum allowed hold_ticks value.
            default_hold_ticks: Default hold_ticks when omitted.
        """

        self._hold_ticks_min = int(hold_ticks_min)
        self._hold_ticks_max = int(hold_ticks_max)
        self._default_hold_ticks = int(default_hold_ticks)
        self._json_pattern = re.compile(r"\{.*\}", re.DOTALL)

    def parse(
        self,
        text: str,
        *,
        legal_moves: Optional[Iterable[str]] = None,
    ) -> RetroParseResult:
        """Parse a macro action from text and optionally validate legality.

        Args:
            text: Raw model output or user input.
            legal_moves: Optional list of legal macro actions.

        Returns:
            The parsed move result.
        """

        raw = text or ""
        stripped = raw.strip()
        if not stripped:
            return RetroParseResult(None, None, raw, "empty_text")

        parsed_move, hold_ticks, parse_error = self._parse_payload(stripped)
        if not parsed_move:
            return RetroParseResult(None, None, raw, parse_error or "invalid_format")

        if legal_moves is not None:
            legal_set = {str(move) for move in legal_moves}
            if parsed_move not in legal_set:
                return RetroParseResult(parsed_move, parsed_move, raw, "illegal_move", hold_ticks)

        return RetroParseResult(parsed_move, parsed_move, raw, None, hold_ticks)

    def build_rethink_prompt(
        self,
        *,
        last_output: str,
        reason: str,
        legal_moves: Sequence[str],
    ) -> str:
        """Build a retry prompt when an illegal action is detected.

        Args:
            last_output: The previous model output.
            reason: Explanation for why the action is invalid.
            legal_moves: List of legal macro actions.

        Returns:
            A formatted prompt for rethinking.
        """

        legal_block = ", ".join(legal_moves)
        return DEFAULT_RETHINK_TEMPLATE.format(
            reason=reason,
            last_output=last_output,
            legal_moves=legal_block,
        )

    def build_action_dict(self, *, player: str, parse_result: RetroParseResult) -> dict[str, Any]:
        """Build a section-15 action dict from a parse result.

        Args:
            player: Player id issuing the action.
            parse_result: Parsed action payload.

        Returns:
            Action dict aligned with section 15.2.
        """

        move = parse_result.move or parse_result.coord or ""
        payload: dict[str, Any] = {"player": str(player), "move": str(move), "raw": str(parse_result.raw)}
        if parse_result.hold_ticks is not None:
            payload["hold_ticks"] = int(parse_result.hold_ticks)
        return payload

    def _parse_payload(self, text: str) -> tuple[Optional[str], Optional[int], Optional[str]]:
        payload = self._extract_json(text)
        move = None
        hold_ticks = None
        if isinstance(payload, dict):
            move = payload.get("move") or payload.get("action")
            hold_value = payload.get("hold_ticks")
            if hold_value is not None:
                try:
                    hold_ticks = int(hold_value)
                except (TypeError, ValueError):
                    return None, None, "invalid_hold_ticks"
        if move is None:
            move = text.splitlines()[-1].strip()
        if not move:
            return None, None, "invalid_format"
        if hold_ticks is None:
            hold_ticks = self._default_hold_ticks
        hold_ticks = max(self._hold_ticks_min, min(self._hold_ticks_max, hold_ticks))
        return str(move), hold_ticks, None

    def _extract_json(self, text: str) -> Optional[dict]:
        matches = list(self._json_pattern.finditer(text))
        if not matches:
            return None
        raw_json = matches[-1].group(0)
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
        return None


__all__ = ["RetroActionParser", "RetroParseResult"]
