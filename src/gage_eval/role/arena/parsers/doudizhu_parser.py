"""Parsing utilities for Doudizhu moves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from gage_eval.registry import registry
from gage_eval.role.arena.games.doudizhu.parsers.doudizhu import DoudizhuMoveParser

DEFAULT_RETHINK_TEMPLATE = (
    "Your previous move could not be processed.\n"
    "Reason: {reason}.\n"
    "Your last output was: '{last_output}'.\n"
    "Instructions:\n"
    "- Output exactly one legal action string from the list below.\n"
    "- Use 'pass' when you choose to skip.\n"
    "Legal moves: {legal_moves}."
)


@dataclass(frozen=True)
class DoudizhuParseResult:
    """Represents the parsed move and any parsing error."""

    move: Optional[str]
    coord: Optional[str]
    raw: str
    error: Optional[str]
    chat_text: Optional[str] = None


@registry.asset(
    "parser_impls",
    "doudizhu_arena_parser_v1",
    desc="Doudizhu move parser for arena text actions",
    tags=("doudizhu", "parser", "arena"),
)
class DoudizhuParser:
    """Parse Doudizhu moves from model output."""

    def __init__(self, *_, **__) -> None:
        """Initialize the parser."""

        self._card_parser = DoudizhuMoveParser()

    def parse(
        self,
        text: str,
        *,
        legal_moves: Optional[Iterable[str]] = None,
    ) -> DoudizhuParseResult:
        """Parse a move from text and optionally validate against legal moves.

        Args:
            text: Raw model output or user input.
            legal_moves: Optional list of legal action strings.

        Returns:
            The parsed move result.
        """

        raw = text or ""
        stripped = raw.strip()
        if not stripped:
            return DoudizhuParseResult(None, None, raw, "empty_text")

        try:
            parsed = self._card_parser.parse(raw)
        except Exception as exc:
            return DoudizhuParseResult(None, None, raw, f"parse_error:{exc}")

        action_text = parsed.action_text
        if legal_moves is not None:
            legal_set = {str(move) for move in legal_moves}
            if action_text not in legal_set:
                return DoudizhuParseResult(action_text, action_text, raw, "illegal_move", parsed.chat_text)

        return DoudizhuParseResult(action_text, action_text, raw, None, parsed.chat_text)

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

        legal_block = ", ".join(legal_moves)
        return DEFAULT_RETHINK_TEMPLATE.format(
            reason=reason,
            last_output=last_output,
            legal_moves=legal_block,
        )


__all__ = ["DoudizhuParseResult", "DoudizhuParser"]
