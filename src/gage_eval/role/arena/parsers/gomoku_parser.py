"""Parsing utilities for Gomoku moves."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Optional, Sequence, Tuple

from gage_eval.registry import registry
from gage_eval.role.arena.games.gomoku.coord_scheme import GomokuCoordCodec, normalize_coord_scheme

DEFAULT_RETHINK_TEMPLATE = (
    "Your previous move could not be processed.\n"
    "Reason: {reason}.\n"
    "Your last output was: '{last_output}'.\n"
    "Instructions:\n"
    "- You must select a valid coordinate from the legal moves list.\n"
    "- Please output ONLY the coordinate (e.g., 'H8') on the last line to ensure it is parsed correctly.\n"
    "Legal moves: {legal_moves}."
)


@dataclass(frozen=True)
class GomokuParseResult:
    """Represents the parsed move and any parsing error."""

    move: Optional[Tuple[int, int]]
    coord: Optional[str]
    raw: str
    error: Optional[str]


@registry.asset(
    "parser_impls",
    "gomoku_v1",
    desc="Gomoku move parser (configurable coordinate scheme)",
    tags=("gomoku", "parser"),
)
class GomokuParser:
    """Parse Gomoku move coordinates from model output."""

    def __init__(self, board_size: int = 15, coord_scheme: str = "A1") -> None:
        """Initialize the parser.

        Args:
            board_size: Board dimension (board_size x board_size).
            coord_scheme: Coordinate scheme identifier (default: A1).
        """

        if board_size < 1:
            raise ValueError("board_size must be positive")
        self.board_size = board_size
        self.coord_scheme = normalize_coord_scheme(coord_scheme)
        self._coord_codec = GomokuCoordCodec(board_size=board_size, coord_scheme=self.coord_scheme)
        self._coord_pattern = self._build_coord_pattern()
        self._pair_pattern = re.compile(r"(\d{1,3})\s*[,\s]+\s*(\d{1,3})")

    def parse(
        self,
        text: str,
        *,
        legal_moves: Optional[Iterable[Tuple[int, int]] | Iterable[str]] = None,
    ) -> GomokuParseResult:
        """Parse a move from text and optionally validate against legal moves.

        Args:
            text: Raw model output or user input.
            legal_moves: Optional list of legal moves (tuples or A1 coords).

        Returns:
            The parsed move result.
        """

        raw = text or ""
        stripped = raw.strip()
        if not stripped:
            return GomokuParseResult(None, None, raw, "empty_text")

        candidates, parse_error = self._parse_candidates(stripped)
        if not candidates:
            return GomokuParseResult(None, None, raw, parse_error or "invalid_format")

        legal_tuple_set, legal_coord_set = self._normalize_legal_moves(legal_moves)
        if legal_tuple_set is not None or legal_coord_set is not None:
            legal_candidates = []
            for move, coord in candidates:
                if legal_tuple_set is not None and move in legal_tuple_set:
                    legal_candidates.append((move, coord))
                elif legal_coord_set is not None and coord in legal_coord_set:
                    legal_candidates.append((move, coord))
            if legal_candidates:
                move, coord = legal_candidates[-1]
                return GomokuParseResult(move, coord, raw, None)
            move, coord = candidates[-1]
            return GomokuParseResult(move, coord, raw, "illegal_move")

        move, coord = candidates[-1]
        return GomokuParseResult(move, coord, raw, None)

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
            legal_moves: List of legal move coordinates.

        Returns:
            A formatted prompt for rethinking.
        """

        legal_block = ", ".join(legal_moves)
        return DEFAULT_RETHINK_TEMPLATE.format(
            reason=reason,
            last_output=last_output,
            legal_moves=legal_block,
        )

    def _parse_candidates(self, text: str) -> Tuple[list[Tuple[Tuple[int, int], str]], Optional[str]]:
        candidates: list[Tuple[Tuple[int, int], str]] = []
        parse_error: Optional[str] = None
        if self.coord_scheme == "ROW_COL":
            for match in self._pair_pattern.finditer(text):
                row = int(match.group(1))
                col = int(match.group(2))
                if not (1 <= row <= self.board_size and 1 <= col <= self.board_size):
                    parse_error = parse_error or "out_of_bounds"
                    continue
                coord = f"{row},{col}"
                candidates.append(((row - 1, col - 1), coord))
            return candidates, parse_error

        for match in self._coord_pattern.finditer(text):
            col_label = match.group(1).upper()
            row = int(match.group(2))
            move, coord, error = self._build_move(col_label, row)
            if error or move is None or coord is None:
                parse_error = parse_error or error
                continue
            candidates.append((move, coord))
        if candidates:
            return candidates, parse_error

        for match in self._pair_pattern.finditer(text):
            row = int(match.group(1))
            col = int(match.group(2))
            if not (1 <= row <= self.board_size and 1 <= col <= self.board_size):
                parse_error = parse_error or "out_of_bounds"
                continue
            try:
                coord = self._coord_codec.index_to_coord(row - 1, col - 1)
            except ValueError:
                parse_error = parse_error or "out_of_bounds"
                continue
            candidates.append(((row - 1, col - 1), coord))
        return candidates, parse_error

    def _build_move(
        self,
        col_label: str,
        row: int,
    ) -> Tuple[Optional[Tuple[int, int]], Optional[str], Optional[str]]:
        if not (1 <= row <= self.board_size):
            return None, None, "out_of_bounds"
        try:
            col_index = self._coord_codec.coord_to_index(f"{col_label}{row}")[1]
        except ValueError:
            return None, None, "out_of_bounds"
        move = (row - 1, col_index)
        coord = f"{col_label}{row}"
        return move, coord, None

    def _build_coord_pattern(self) -> re.Pattern[str]:
        if self.coord_scheme == "AA1":
            return re.compile(r"([A-Za-z]{1,3})\s*([1-9]\d{0,2})")
        return re.compile(r"([A-Za-z])\s*([1-9]\d{0,2})")

    @staticmethod
    def _normalize_row_col(coord: str) -> str:
        match = re.match(r"^\s*(\d{1,3})\s*[,:\s]+\s*(\d{1,3})\s*$", coord)
        if not match:
            return coord.strip()
        return f"{int(match.group(1))},{int(match.group(2))}"

    def _normalize_legal_moves(
        self,
        legal_moves: Optional[Iterable[Tuple[int, int]] | Iterable[str]],
    ) -> Tuple[Optional[set[Tuple[int, int]]], Optional[set[str]]]:
        if legal_moves is None:
            return None, None
        moves_list = list(legal_moves)
        if not moves_list:
            return None, None
        if all(isinstance(move, str) for move in moves_list):
            if self.coord_scheme == "ROW_COL":
                return None, {self._normalize_row_col(str(move)) for move in moves_list}
            return None, {move.upper() for move in moves_list}
        if all(isinstance(move, tuple) and len(move) == 2 for move in moves_list):
            return {move for move in moves_list if isinstance(move, tuple)}, None
        raise ValueError("legal_moves must be coordinates or row/col tuples")


@registry.asset(
    "parser_impls",
    "grid_parser_v1",
    desc="Grid game move parser (configurable coordinate scheme)",
    tags=("grid", "parser"),
)
class GridParser(GomokuParser):
    """Parse grid game move coordinates from model output."""
