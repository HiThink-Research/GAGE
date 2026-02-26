"""Parser for ViZDoom-style discrete action outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Iterable, Optional, Sequence

from gage_eval.registry import registry
from gage_eval.role.arena.parsers.vizdoom_action_codec import VizDoomActionCodec

DEFAULT_RETHINK_TEMPLATE = (
    "Your previous action could not be parsed.\n"
    "Reason: {reason}.\n"
    "Your last output was: '{last_output}'.\n"
    "Instructions:\n"
    "- You must select a valid action id from the legal actions list.\n"
    "- Output ONLY the action id as an integer.\n"
    "Legal actions: {legal_moves}."
)


@dataclass(frozen=True)
class VizDoomParseResult:
    """Represents the parsed ViZDoom action and any parsing error."""

    move: Optional[int]
    coord: Optional[str]
    raw: str
    error: Optional[str]


@registry.asset(
    "parser_impls",
    "vizdoom_parser_v1",
    desc="ViZDoom parser for discrete action ids",
    tags=("vizdoom", "parser"),
)
class VizDoomParser:
    """Parse discrete action ids from raw model output."""

    def __init__(
        self,
        *,
        action_labels: Optional[Sequence[str]] = None,
        action_mapping: Optional[dict[str, int]] = None,
        default_action: int = 0,
        **_: object,
    ) -> None:
        """Initialize the parser.

        Args:
            action_labels: Optional ordered labels for action ids.
            action_mapping: Optional explicit label-to-id mapping.
            default_action: Fallback action id when parsing fails.
        """

        self._codec = VizDoomActionCodec(
            action_labels=action_labels,
            action_mapping=action_mapping,
            default_action=default_action,
        )

    def parse(
        self,
        text: str,
        *,
        legal_moves: Optional[Iterable[int] | Iterable[str]] = None,
    ) -> VizDoomParseResult:
        """Parse a move from text and optionally validate against legal moves.

        Args:
            text: Raw model output or user input.
            legal_moves: Optional list of legal moves.

        Returns:
            Parsed action result.
        """

        raw = text or ""
        stripped = raw.strip()
        if not stripped:
            return VizDoomParseResult(None, None, raw, "empty_text")

        parsed = self._parse_action_value(stripped)
        if parsed is None:
            return VizDoomParseResult(None, None, raw, "invalid_format")

        move = self._codec.encode(parsed)
        if legal_moves is not None:
            legal_set = {str(item) for item in legal_moves}
            if str(move) not in legal_set and str(parsed) not in legal_set:
                return VizDoomParseResult(move, str(move), raw, "illegal_move")

        return VizDoomParseResult(move, str(move), raw, None)

    def build_rethink_prompt(
        self,
        *,
        last_output: str,
        reason: str,
        legal_moves: Sequence[str],
    ) -> str:
        """Build a retry prompt for invalid actions.

        Args:
            last_output: The previous model output.
            reason: Explanation for why the action is invalid.
            legal_moves: List of legal action ids.

        Returns:
            A formatted prompt for rethinking.
        """

        legal_block = ", ".join(legal_moves)
        return DEFAULT_RETHINK_TEMPLATE.format(
            reason=reason,
            last_output=last_output,
            legal_moves=legal_block,
        )

    def _parse_action_value(self, text: str) -> Optional[object]:
        if text.startswith("{") and text.endswith("}"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                for key in ("action", "move", "id"):
                    if key in payload:
                        return payload[key]
        match = re.findall(r"-?\d+", text)
        if match:
            return match[-1]
        return text
