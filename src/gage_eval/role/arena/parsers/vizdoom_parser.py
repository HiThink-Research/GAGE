"""Parser for ViZDoom-style discrete action outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable, Optional, Sequence

from gage_eval.registry import registry
from gage_eval.role.arena.parsers.vizdoom_action_codec import VizDoomActionCodec

DEFAULT_RETHINK_TEMPLATE = (
    "Your previous action could not be parsed.\n"
    "Reason: {reason}.\n"
    "Your last output was: '{last_output}'.\n"
    "Instructions:\n"
    "- You must select a valid action id from the legal actions list.\n"
    "- Output exactly two lines:\n"
    "  Action: <action_id>\n"
    "  Reason: <short reason>\n"
    "Legal actions: {legal_moves}."
)


@dataclass(frozen=True)
class VizDoomParseResult:
    """Represents the parsed ViZDoom action and any parsing error."""

    move: Optional[int]
    coord: Optional[str]
    raw: str
    error: Optional[str]
    reason: Optional[str] = None


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

        # STEP 1: Extract action candidate and optional reason text.
        parsed, parsed_reason = self._extract_action_and_reason(stripped)
        if parsed is None:
            return VizDoomParseResult(None, None, raw, "invalid_format", parsed_reason)

        # STEP 2: Validate parsed action with legal moves and recover from noisy reasons.
        move = self._codec.encode(parsed)
        if legal_moves is not None:
            legal_list = [str(item) for item in legal_moves]
            legal_set = set(legal_list)
            if str(move) not in legal_set and str(parsed) not in legal_set:
                recovered = self._recover_action_from_legal(stripped, legal_list)
                if recovered is not None:
                    parsed = recovered
                    move = self._codec.encode(recovered)
            if str(move) not in legal_set and str(parsed) not in legal_set:
                return VizDoomParseResult(move, str(move), raw, "illegal_move", parsed_reason)

        return VizDoomParseResult(move, str(move), raw, None, parsed_reason)

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

    def _extract_action_and_reason(self, text: str) -> tuple[Optional[object], Optional[str]]:
        """Extract action payload and optional reason string from model output."""

        if text.startswith("{") and text.endswith("}"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                action = self._extract_action_from_payload(payload)
                reason = self._extract_reason_from_payload(payload)
                if action is not None:
                    return action, reason

        reason = self._extract_reason_text(text)
        action_line_value = self._extract_action_from_labeled_line(text)
        if action_line_value is not None:
            return action_line_value, reason

        first_line_value = self._extract_action_from_first_line(text)
        if first_line_value is not None:
            return first_line_value, reason

        return self._parse_action_value(text), reason

    @staticmethod
    def _extract_action_from_payload(payload: dict[str, Any]) -> Optional[object]:
        """Extract action value from JSON payload keys."""

        for key in ("action", "move", "id"):
            if key in payload:
                return payload[key]
        return None

    @staticmethod
    def _extract_reason_from_payload(payload: dict[str, Any]) -> Optional[str]:
        """Extract short reason text from JSON payload keys."""

        for key in ("reason", "rationale", "analysis", "explanation", "why"):
            value = payload.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    def _extract_action_from_labeled_line(self, text: str) -> Optional[object]:
        """Extract action value from lines like `Action: 2`."""

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            match = re.match(r"^(?:action|move|id)\s*[:=]\s*(.+)$", stripped, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip()
            if not candidate:
                continue
            parsed = self._parse_action_value(candidate)
            if parsed is not None:
                return parsed
        return None

    def _extract_action_from_first_line(self, text: str) -> Optional[object]:
        """Extract action value when the first line is a compact action token."""

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parsed = self._parse_action_value(stripped)
            if parsed is not None:
                return parsed
            break
        return None

    def _recover_action_from_legal(
        self,
        text: str,
        legal_moves: Sequence[str],
    ) -> Optional[object]:
        """Recover a legal action candidate when free-form reason text is noisy."""

        if not legal_moves:
            return None
        legal_lookup = {str(item).strip().lower(): str(item) for item in legal_moves}

        candidates: list[str] = []
        action_line = self._extract_action_from_labeled_line(text)
        if action_line is not None:
            candidates.append(str(action_line))
        for token in re.findall(r"-?\d+", text):
            candidates.append(token)
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
            candidates.append(token)

        for candidate in candidates:
            normalized = candidate.strip().lower()
            if normalized in legal_lookup:
                return legal_lookup[normalized]
            encoded = self._codec.encode(candidate)
            if str(encoded) in legal_lookup:
                return str(encoded)
        return None

    def _extract_reason_text(self, text: str) -> Optional[str]:
        """Extract reason text from common free-form response patterns."""

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None

        for line in lines:
            match = re.match(
                r"^(?:reason|rationale|analysis|explanation|why)\s*[:=\-]\s*(.+)$",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                parsed = match.group(1).strip()
                return parsed or None

        if len(lines) > 1:
            parsed = " ".join(lines[1:]).strip()
            return parsed or None
        return None

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
