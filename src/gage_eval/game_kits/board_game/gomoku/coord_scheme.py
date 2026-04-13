"""Coordinate scheme helpers for Gomoku boards."""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

_ROW_COL_PATTERN = re.compile(r"^\s*(\d{1,3})\s*[,:\s]+\s*(\d{1,3})\s*$")
_A1_PATTERN = re.compile(r"^\s*([A-Za-z]+)\s*(\d{1,3})\s*$")


def normalize_coord_scheme(value: str) -> str:
    """Normalize coordinate scheme identifiers."""

    normalized = str(value or "A1").strip().upper()
    if normalized in {"ROW_COL", "ROWCOL", "ROW-COL"}:
        return "ROW_COL"
    if normalized in {"AA1", "A1"}:
        return normalized
    return "A1"


class GomokuCoordCodec:
    """Convert between row/col indices and coordinate strings."""

    def __init__(self, board_size: int, coord_scheme: str = "A1") -> None:
        self.board_size = int(board_size)
        self.coord_scheme = normalize_coord_scheme(coord_scheme)
        if self.board_size < 1:
            raise ValueError("board_size must be positive")
        if self.coord_scheme == "A1" and self.board_size > 26:
            raise ValueError("A1 coordinate scheme supports up to 26 columns")

    def column_labels(self) -> List[str]:
        if self.coord_scheme == "ROW_COL":
            return [str(idx + 1) for idx in range(self.board_size)]
        return [index_to_letters(idx) for idx in range(self.board_size)]

    def index_to_coord(self, row: int, col: int) -> str:
        if not self._in_bounds(row, col):
            raise ValueError("row/col out of bounds")
        if self.coord_scheme == "ROW_COL":
            return f"{row + 1},{col + 1}"
        col_label = index_to_letters(col)
        return f"{col_label}{row + 1}"

    def coord_to_index(self, coord: str) -> Tuple[int, int]:
        if self.coord_scheme == "ROW_COL":
            match = _ROW_COL_PATTERN.match(coord or "")
            if not match:
                raise ValueError("coord format invalid")
            row = int(match.group(1))
            col = int(match.group(2))
            if not (1 <= row <= self.board_size and 1 <= col <= self.board_size):
                raise ValueError("coord out of bounds")
            return row - 1, col - 1

        match = _A1_PATTERN.match(coord or "")
        if not match:
            raise ValueError("coord format invalid")
        col_label = match.group(1).upper()
        row = int(match.group(2))
        if not (1 <= row <= self.board_size):
            raise ValueError("coord out of bounds")
        col_index = letters_to_index(col_label)
        if not (0 <= col_index < self.board_size):
            raise ValueError("coord out of bounds")
        if self.coord_scheme == "A1" and len(col_label) != 1:
            raise ValueError("coord format invalid")
        return row - 1, col_index

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.board_size and 0 <= col < self.board_size


def index_to_letters(index: int) -> str:
    """Convert a zero-based index to spreadsheet-style letters."""

    if index < 0:
        raise ValueError("index must be non-negative")
    value = index + 1
    result = ""
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def letters_to_index(label: str) -> int:
    """Convert spreadsheet-style letters to zero-based index."""

    if not label or not label.isalpha():
        raise ValueError("label must be letters")
    total = 0
    for char in label.upper():
        total = total * 26 + (ord(char) - ord("A") + 1)
    return total - 1


def normalize_column_labels(labels: Iterable[str]) -> List[str]:
    """Normalize column labels for display."""

    return [str(label).strip().upper() for label in labels]
