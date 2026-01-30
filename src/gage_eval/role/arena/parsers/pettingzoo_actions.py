"""Discrete action codec and parser for PettingZoo/Gymnasium-style environments."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Optional, Sequence

from gage_eval.registry import registry

DEFAULT_RETHINK_TEMPLATE = (
    "Your previous action could not be processed.\n"
    "Reason: {reason}.\n"
    "Your last output was: '{last_output}'.\n"
    "Instructions:\n"
    "- You must select a valid action from the legal moves list.\n"
    "- Output ONLY the action id or action label on the last line.\n"
    "Legal moves: {legal_moves}."
)


@dataclass(frozen=True)
class DiscreteActionParseResult:
    """Parsed discrete action output."""

    action_id: Optional[int]
    coord: Optional[str]
    raw: str
    error: Optional[str]


class DiscreteActionCodec:
    """Encode/decode discrete actions with optional labels."""

    def __init__(self, action_labels: Optional[Sequence[str]] = None) -> None:
        """Initialize the codec.

        Args:
            action_labels: Optional ordered list of action labels.
        """

        self._action_labels = [str(label) for label in action_labels or []]
        self._label_lookup = {
            self._normalize_token(label): idx for idx, label in enumerate(self._action_labels)
        }

    def legal_moves(self, action_space: Any, *, action_mask: Optional[Sequence[int]] = None) -> list[str]:
        """Return legal move strings for the provided action space.

        Args:
            action_space: Discrete action space with `n` or `__len__`.
            action_mask: Optional mask of legal actions (1 = legal).

        Returns:
            List of legal move strings.
        """

        count = self._resolve_action_count(action_space)
        labels = self._resolve_labels(count)
        if action_mask is None:
            return labels
        resolved_mask = list(action_mask)
        if len(resolved_mask) != count:
            return labels
        return [label for idx, label in enumerate(labels) if resolved_mask[idx]]

    def encode(
        self,
        move: str | int,
        *,
        action_space: Any,
        action_mask: Optional[Sequence[int]] = None,
    ) -> int:
        """Encode a move into an action id.

        Args:
            move: Action label or id string.
            action_space: Discrete action space with `n` or `contains`.
            action_mask: Optional mask of legal actions.

        Returns:
            Encoded action id.

        Raises:
            ValueError: If the action is invalid.
        """

        action_id = self._resolve_action_id(move)
        if action_id is None:
            raise ValueError("invalid_action")
        if not self._is_in_action_space(action_id, action_space):
            raise ValueError("out_of_bounds")
        if action_mask is not None:
            resolved_mask = list(action_mask)
            if action_id < len(resolved_mask) and not resolved_mask[action_id]:
                raise ValueError("illegal_action")
        return action_id

    def decode(self, action_id: int) -> str:
        """Decode an action id into a display label.

        Args:
            action_id: Encoded action identifier.

        Returns:
            Human-readable action label or numeric string.
        """

        if 0 <= action_id < len(self._action_labels):
            return self._action_labels[action_id]
        return str(action_id)

    def _resolve_action_id(self, move: str | int) -> Optional[int]:
        if isinstance(move, int):
            return move
        token = str(move).strip()
        if not token:
            return None
        normalized = self._normalize_token(token)
        if normalized in self._label_lookup:
            return self._label_lookup[normalized]
        match = re.search(r"-?\d+", token)
        if not match:
            return None
        try:
            return int(match.group(0))
        except ValueError:
            return None

    @staticmethod
    def _resolve_action_count(action_space: Any) -> int:
        if hasattr(action_space, "n"):
            return int(action_space.n)
        try:
            return int(len(action_space))
        except Exception as exc:
            raise ValueError("action_space must define size") from exc

    def _resolve_labels(self, count: int) -> list[str]:
        if self._action_labels and len(self._action_labels) == count:
            return list(self._action_labels)
        return [str(idx) for idx in range(count)]

    @staticmethod
    def _normalize_token(token: str) -> str:
        return re.sub(r"\s+", "", token.strip().lower())

    @staticmethod
    def _is_in_action_space(action_id: int, action_space: Any) -> bool:
        if hasattr(action_space, "contains"):
            try:
                return bool(action_space.contains(action_id))
            except Exception:
                return False
        count = DiscreteActionCodec._resolve_action_count(action_space)
        return 0 <= action_id < count


@registry.asset(
    "parser_impls",
    "discrete_action_parser_v1",
    desc="Discrete action parser (numeric or labeled actions)",
    tags=("parser", "action"),
)
class DiscreteActionParser:
    """Parse discrete action selections from model output."""

    def __init__(
        self,
        action_labels: Optional[Sequence[str]] = None,
        board_size: Optional[int] = None,
        coord_scheme: Optional[str] = None,
        **_: object,
    ) -> None:
        """Initialize the parser.

        Args:
            action_labels: Optional ordered action labels.
            board_size: Unused board size (accepted for arena parser compatibility).
            coord_scheme: Unused coordinate scheme (accepted for arena parser compatibility).
        """

        self._action_labels = [str(label) for label in action_labels or []]
        self._normalized_labels = [self._normalize_token(label) for label in self._action_labels]
        self._int_pattern = re.compile(r"-?\d+")

    def parse(
        self,
        text: str,
        *,
        legal_moves: Optional[Iterable[str]] = None,
    ) -> DiscreteActionParseResult:
        """Parse an action from raw text and validate against legal moves.

        Args:
            text: Raw model output.
            legal_moves: Optional list of legal move labels.

        Returns:
            Parsed action result.
        """

        raw = text or ""
        stripped = raw.strip()
        if not stripped:
            return DiscreteActionParseResult(None, None, raw, "empty_text")

        legal_list = [str(move) for move in legal_moves or []]
        legal_lookup = {self._normalize_token(move): move for move in legal_list}

        # STEP 1: Prefer direct/legal matches when provided.
        candidate = self._select_from_legal(stripped, legal_list, legal_lookup)
        if candidate is None:
            candidate = self._select_numeric(stripped)
        if candidate is None:
            candidate = self._select_label(stripped)

        if candidate is None:
            return DiscreteActionParseResult(None, None, raw, "invalid_format")

        if legal_lookup and self._normalize_token(candidate) not in legal_lookup:
            return DiscreteActionParseResult(None, None, raw, "illegal_move")

        resolved = legal_lookup.get(self._normalize_token(candidate), candidate)
        action_id = self._resolve_action_id(resolved)
        return DiscreteActionParseResult(action_id, resolved, raw, None)

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
            legal_moves: List of legal move labels.

        Returns:
            A formatted prompt for rethinking.
        """

        legal_block = ", ".join(legal_moves)
        return DEFAULT_RETHINK_TEMPLATE.format(
            reason=reason,
            last_output=last_output,
            legal_moves=legal_block,
        )

    def _select_from_legal(
        self,
        stripped: str,
        legal_list: Sequence[str],
        legal_lookup: dict[str, str],
    ) -> Optional[str]:
        if not legal_list:
            return None
        normalized = self._normalize_token(stripped)
        if normalized in legal_lookup:
            return legal_lookup[normalized]
        lowered = stripped.lower()
        best_match = None
        best_index = -1
        for move in legal_list:
            token = move.lower()
            idx = lowered.find(token)
            if idx >= 0 and idx >= best_index:
                best_index = idx
                best_match = move
        return best_match

    def _select_numeric(self, stripped: str) -> Optional[str]:
        matches = list(self._int_pattern.finditer(stripped))
        if not matches:
            return None
        return matches[-1].group(0)

    def _select_label(self, stripped: str) -> Optional[str]:
        lowered = self._normalize_token(stripped)
        best_label = None
        for label, normalized in zip(self._action_labels, self._normalized_labels):
            if normalized and normalized in lowered:
                best_label = label
        return best_label

    @staticmethod
    def _normalize_token(token: str) -> str:
        return re.sub(r"\s+", "", token.strip().lower())

    def _resolve_action_id(self, token: str) -> Optional[int]:
        normalized = self._normalize_token(token)
        for idx, label in enumerate(self._normalized_labels):
            if normalized == label:
                return idx
        match = self._int_pattern.search(token)
        if not match:
            return None
        try:
            return int(match.group(0))
        except ValueError:
            return None


__all__ = ["DiscreteActionCodec", "DiscreteActionParser", "DiscreteActionParseResult"]
