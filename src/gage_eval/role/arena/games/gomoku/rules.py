"""Rule helpers for Gomoku."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple


@dataclass(frozen=True)
class GomokuWinInfo:
    """Describe a winning line and its direction."""

    line: Sequence[Tuple[int, int]]
    direction: str
    line_length: int


_DIRECTION_DELTAS: Dict[str, Tuple[int, int]] = {
    "horizontal": (0, 1),
    "vertical": (1, 0),
    "diagonal": (1, 1),
    "anti_diagonal": (1, -1),
}


@dataclass(frozen=True)
class GomokuRuleEngine:
    """Checks Gomoku win conditions on a board."""

    win_len: int
    empty_cell: str = "."
    rule_profile: str = "freestyle"
    win_directions: Sequence[str] = ("horizontal", "vertical", "diagonal", "anti_diagonal")

    def check_win(self, board: Sequence[Sequence[str]], row: int, col: int) -> bool:
        """Return True if the last move creates a win.

        Args:
            board: Current board state.
            row: Row index of the last move.
            col: Column index of the last move.

        Returns:
            True if the move wins the game.
        """

        return self.find_win_info(board, row, col) is not None

    def find_win_line(
        self,
        board: Sequence[Sequence[str]],
        row: int,
        col: int,
    ) -> list[tuple[int, int]]:
        """Return the contiguous winning line that includes the last move.

        Args:
            board: Current board state.
            row: Row index of the last move.
            col: Column index of the last move.

        Returns:
            List of (row, col) pairs for a winning line, or empty if none.
        """

        info = self.find_win_info(board, row, col)
        return list(info.line) if info else []

    def find_win_info(
        self,
        board: Sequence[Sequence[str]],
        row: int,
        col: int,
    ) -> GomokuWinInfo | None:
        """Return the winning line info that includes the last move."""

        token = board[row][col]
        if token == self.empty_cell:
            return None
        directions = self._resolve_directions(self.win_directions)
        size = len(board)
        for direction_name, (delta_row, delta_col) in directions:
            line: list[tuple[int, int]] = [(row, col)]
            cur_row = row - delta_row
            cur_col = col - delta_col
            while (
                0 <= cur_row < size
                and 0 <= cur_col < size
                and board[cur_row][cur_col] == token
            ):
                line.insert(0, (cur_row, cur_col))
                cur_row -= delta_row
                cur_col -= delta_col
            cur_row = row + delta_row
            cur_col = col + delta_col
            while (
                0 <= cur_row < size
                and 0 <= cur_col < size
                and board[cur_row][cur_col] == token
            ):
                line.append((cur_row, cur_col))
                cur_row += delta_row
                cur_col += delta_col
            line_length = len(line)
            if line_length < self.win_len:
                continue
            if not self._is_valid_line_length(line_length):
                continue
            return GomokuWinInfo(line=line, direction=direction_name, line_length=line_length)
        return None

    def _count_direction(
        self,
        board: Sequence[Sequence[str]],
        row: int,
        col: int,
        delta_row: int,
        delta_col: int,
        token: str,
    ) -> int:
        count = 0
        cur_row = row + delta_row
        cur_col = col + delta_col
        size = len(board)
        while 0 <= cur_row < size and 0 <= cur_col < size and board[cur_row][cur_col] == token:
            count += 1
            cur_row += delta_row
            cur_col += delta_col
        return count

    def _is_valid_line_length(self, line_length: int) -> bool:
        profile = str(self.rule_profile or "freestyle").lower()
        if profile in {"renju", "exact"}:
            return line_length == self.win_len
        return line_length >= self.win_len

    @staticmethod
    def _resolve_directions(directions: Iterable[str]) -> list[tuple[str, tuple[int, int]]]:
        resolved: list[tuple[str, tuple[int, int]]] = []
        for value in directions:
            key = str(value).strip().lower()
            if key in {"h", "horizontal"}:
                name = "horizontal"
            elif key in {"v", "vertical"}:
                name = "vertical"
            elif key in {"diag", "diagonal"}:
                name = "diagonal"
            elif key in {"anti", "anti_diagonal", "anti-diagonal"}:
                name = "anti_diagonal"
            else:
                continue
            resolved.append((name, _DIRECTION_DELTAS[name]))
        if not resolved:
            resolved = [
                (name, delta) for name, delta in _DIRECTION_DELTAS.items()
            ]
        return resolved
