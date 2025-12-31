"""Tic-Tac-Toe board renderer for HTML-based visualization."""

from __future__ import annotations

import html
import re
from typing import Any, Optional, Sequence

from gage_eval.registry import registry
from gage_eval.role.arena.games.gomoku.coord_scheme import GomokuCoordCodec, normalize_coord_scheme
from gage_eval.role.arena.visualizers.utils import build_board_interaction_js

EMPTY_CELL = "."

TICTACTOE_BOARD_CSS = """
#gomoku-shell,
.gomoku-shell {
  font-family: 'Fredoka', 'Baloo 2', 'Trebuchet MS', sans-serif;
  color: #2f2f2f;
  -webkit-font-smoothing: antialiased;
}

/* Global Fixes */
.toast-wrap,
.toast,
.toast-container,
.gradio-error,
.error,
.error-message,
.error-box,
.error-panel,
.notification,
.alert,
[role="alert"],
[aria-live="assertive"],
[aria-live="polite"] { display: none !important; }
.gradio-container { overflow-y: scroll !important; }

/* Layout */
#gomoku-layout {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  align-items: stretch;
}
#gomoku-left-panel,
#gomoku-right-panel {
  background: #ffffff;
  border-radius: 16px;
  padding: 16px;
  border: 1px solid #f0e7d7;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
}
#gomoku-left-panel {
  flex: 2;
  min-width: 320px;
  display: flex;
  flex-direction: column;
  align-items: center;
}
#gomoku-right-panel {
  flex: 1;
  min-width: 280px;
  display: flex;
  flex-direction: column;
}

/* Player Cards */
.gomoku-players {
  display: flex;
  gap: 16px;
  margin-bottom: 20px;
  width: 100%;
}
.gomoku-player {
  flex: 1;
  padding: 16px 18px;
  border-radius: 14px;
  border: 2px solid #f4d2a6;
  background: linear-gradient(160deg, #fff5e8, #fffdf9);
  position: relative;
  overflow: hidden;
  box-shadow: 0 8px 12px rgba(0, 0, 0, 0.05);
}
.gomoku-player::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 6px;
  background: #f8b26a;
}
.gomoku-player.black {
  border-color: #ffb199;
  background: linear-gradient(160deg, #ffe6db, #fff4ed);
}
.gomoku-player.black::before { background: #ff8a65; }
.gomoku-player.white {
  border-color: #9adbe8;
  background: linear-gradient(160deg, #e6f9ff, #f7feff);
}
.gomoku-player.white::before { background: #4dd0e1; }
.gomoku-player.neutral {
  border-style: dashed;
  opacity: 0.8;
}
.player-label {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #7a6a58;
  margin-bottom: 6px;
  padding-left: 10px;
}
.player-name {
  font-size: 20px;
  font-weight: 700;
  color: #2b2b2b;
  padding-left: 10px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.gomoku-player.thinking {
  transform: translateY(-2px);
  box-shadow: 0 12px 20px rgba(0, 0, 0, 0.12);
}
.gomoku-player.thinking::after {
  content: "";
  position: absolute;
  left: 12px;
  right: 12px;
  bottom: 10px;
  height: 4px;
  border-radius: 999px;
  background-size: 220% 100%;
  animation: ttt-thinking 1.6s linear infinite;
}
.gomoku-player.black.thinking::after {
  background-image: linear-gradient(90deg, rgba(255, 138, 101, 0.2), rgba(255, 138, 101, 0.9), rgba(255, 138, 101, 0.2));
}
.gomoku-player.white.thinking::after {
  background-image: linear-gradient(90deg, rgba(77, 208, 225, 0.2), rgba(77, 208, 225, 0.9), rgba(77, 208, 225, 0.2));
}
@keyframes ttt-thinking {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
.status-badge {
  display: none;
  position: absolute;
  top: 12px;
  right: 12px;
}
.gomoku-player.thinking .status-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  background: rgba(255, 255, 255, 0.8);
  border: 1px solid rgba(0, 0, 0, 0.06);
  padding: 4px 10px;
  border-radius: 999px;
}
.thinking-text {
  font-size: 10px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #7a6a58;
}
.timer-text {
  font-family: 'IBM Plex Mono', 'Menlo', 'Monaco', monospace;
  font-size: 12px;
  font-weight: 700;
  color: #ff7043;
}

/* Board */
#gomoku-board-container {
  width: 100%;
  display: flex;
  justify-content: center;
}
.tictactoe-board-wrapper {
  background: linear-gradient(180deg, #fff4e1, #ffe4c6);
  border: 4px solid #c97f4f;
  border-radius: 20px;
  padding: 16px;
  box-shadow: 0 12px 30px rgba(131, 79, 41, 0.25);
  position: relative;
  display: inline-block;
  margin: 0 auto;
}
.tictactoe-coords-row {
  display: grid;
  grid-template-columns: 32px repeat(var(--tt-size), var(--tt-cell)) 32px;
  align-items: center;
  margin-bottom: 6px;
  font-size: 12px;
  color: #7b5a3a;
  font-weight: 700;
}
.tictactoe-coords-col {
  display: grid;
  grid-template-rows: repeat(var(--tt-size), var(--tt-cell));
  align-items: center;
  font-size: 12px;
  color: #7b5a3a;
  font-weight: 700;
}
.tictactoe-coord-label {
  display: flex;
  align-items: center;
  justify-content: center;
}
.tictactoe-board-middle {
  display: flex;
  align-items: center;
  gap: 6px;
}
.tictactoe-grid {
  display: grid;
  grid-template-columns: repeat(var(--tt-size), var(--tt-cell));
  grid-template-rows: repeat(var(--tt-size), var(--tt-cell));
  gap: 8px;
  background: #c97f4f;
  padding: 8px;
  border-radius: 16px;
}
.tictactoe-cell {
  width: var(--tt-cell);
  height: var(--tt-cell);
  background: #fffaf2;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  position: relative;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
  box-shadow: inset 0 2px 0 rgba(255, 255, 255, 0.8);
}
.tictactoe-board-wrapper[data-interactive="true"] .tictactoe-cell:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(201, 127, 79, 0.35);
}
.tictactoe-cell.last-move {
  outline: 3px solid #ffb74d;
  outline-offset: 2px;
}
.tictactoe-cell.win-line {
  box-shadow: 0 0 0 3px #8bc34a, 0 0 16px rgba(139, 195, 74, 0.6);
}
.tictactoe-mark {
  font-size: calc(var(--tt-cell) * 0.6);
  font-weight: 900;
  letter-spacing: -1px;
  animation: ttt-pop 0.2s ease-out;
}
.tictactoe-mark.x {
  color: #ff7043;
  text-shadow: 2px 2px 0 rgba(255, 112, 67, 0.2);
}
.tictactoe-mark.o {
  color: #26c6da;
  text-shadow: 2px 2px 0 rgba(38, 198, 218, 0.2);
}
@keyframes ttt-pop {
  0% { transform: scale(0.6); opacity: 0.3; }
  100% { transform: scale(1); opacity: 1; }
}

/* Output Panel */
.gomoku-output-label {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px;
  border-bottom: 1px solid #f2e8d8;
}
#gomoku-output-box textarea {
  font-family: 'IBM Plex Mono', 'Menlo', 'Monaco', monospace;
  font-size: 13px;
  line-height: 1.6;
  color: #3a2f26;
  background: #fffdf9;
  border: 1px solid #f2e8d8;
  border-radius: 8px;
  padding: 12px;
  min-height: 600px !important;
}
.player-pill {
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
}
.player-pill.black { background: #ff7043; color: #fff; }
.player-pill.white { background: #26c6da; color: #fff; }
.move-pill {
  background: #ffe0b2;
  color: #6d4c41;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 700;
}

/* Hidden Controls */
#gomoku-move-submit {
  flex: 0 0 auto;
  opacity: 0;
  width: 0;
  height: 0;
  margin: 0;
  padding: 0;
  overflow: hidden;
}
#gomoku-move-submit button {
  width: 0;
  height: 0;
  min-width: 0;
  min-height: 0;
  padding: 0;
  border: 0;
}
#gomoku-finish-button {
  width: 100%;
  margin-top: 12px;
  background: #ffb74d;
  color: #3a2f26 !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  border: 2px solid #f6a03a !important;
}
.finish-pulse {
  animation: ttt-finish-pulse 1.8s infinite;
  background: #ff7043 !important;
  color: #fff !important;
  border-color: #ff7043 !important;
  box-shadow: 0 0 0 0 rgba(255, 112, 67, 0.6);
}
@keyframes ttt-finish-pulse {
  0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 112, 67, 0.55); }
  60% { transform: scale(1.03); box-shadow: 0 0 0 12px rgba(255, 112, 67, 0); }
  100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 112, 67, 0); }
}
#gomoku-refresh-button {
  display: none !important;
}
"""


@registry.asset(
    "renderer_impls",
    "tictactoe_board_v1",
    desc="Tic-Tac-Toe board renderer (HTML/CSS grid)",
    tags=("tictactoe", "renderer"),
)
class TicTacToeBoardRenderer:
    """Render and parse Tic-Tac-Toe boards from text snapshots."""

    def __init__(self, board_size: int, coord_scheme: str = "ROW_COL", show_coords: bool = True) -> None:
        self._board_size = int(board_size)
        self._coord_scheme = normalize_coord_scheme(coord_scheme)
        self._coord_codec = GomokuCoordCodec(self._board_size, self._coord_scheme)
        self._grid = self._default_grid()
        self._raw_text = self._render_text(self._grid)
        self._last_move: Optional[str] = None
        self._winning_line: set[str] = set()
        self._show_coords = bool(show_coords)

    def update(
        self,
        board_text: str,
        *,
        last_move: Optional[str] = None,
        winning_line: Optional[Sequence[str]] = None,
    ) -> None:
        """Update the renderer with a new text snapshot."""

        if not board_text:
            return
        self._raw_text = board_text
        self._last_move = self._normalize_coord(last_move)
        self._winning_line = self._normalize_coords(winning_line)
        parsed = self._parse_board_text(board_text)
        if parsed is None:
            return
        if len(parsed) != self._board_size:
            self._board_size = len(parsed)
            self._coord_codec = GomokuCoordCodec(self._board_size, self._coord_scheme)
        self._grid = parsed

    def resize(self, board_size: int) -> None:
        """Resize the renderer grid if needed."""

        size = int(board_size)
        if size < 1 or size == self._board_size:
            return
        self._board_size = size
        self._coord_codec = GomokuCoordCodec(self._board_size, self._coord_scheme)
        self._grid = self._default_grid()
        self._raw_text = self._render_text(self._grid)

    def set_coord_scheme(self, coord_scheme: str) -> None:
        """Update the renderer coordinate scheme."""

        normalized = normalize_coord_scheme(coord_scheme)
        if normalized == self._coord_scheme:
            return
        self._coord_scheme = normalized
        self._coord_codec = GomokuCoordCodec(self._board_size, self._coord_scheme)
        self._raw_text = self._render_text(self._grid)

    def raw_text(self) -> str:
        """Return the latest raw board text."""

        return self._raw_text

    def get_css(self) -> str:
        """Return CSS for the Tic-Tac-Toe board."""

        return TICTACTOE_BOARD_CSS

    def build_interaction_js(
        self,
        *,
        board_container_id: str,
        move_input_id: str,
        submit_button_id: str,
        enable_click: bool,
        refresh_button_id: str,
        refresh_interval_ms: int,
    ) -> str:
        """Return JS for click-to-move interactions."""

        return build_board_interaction_js(
            board_container_id,
            move_input_id,
            submit_button_id,
            enable_click=enable_click,
            refresh_button_id=refresh_button_id,
            refresh_interval_ms=refresh_interval_ms,
        )

    def render_html(self, *, interactive: bool) -> str:
        """Render the current grid into styled HTML markup."""

        col_labels = self._column_labels() if self._show_coords else []
        cell_size = self._resolve_cell_size()
        wrapper_style = f'style="--tt-size:{self._board_size}; --tt-cell:{cell_size}px;"'

        def render_col_labels_row() -> str:
            if not col_labels:
                return ""
            html_parts = ['<div class="tictactoe-coords-row">', '<div></div>']
            for label in col_labels:
                html_parts.append(f'<div class="tictactoe-coord-label">{label}</div>')
            html_parts.append('<div></div>')
            html_parts.append("</div>")
            return "".join(html_parts)

        def render_grid() -> str:
            grid_blocks = ['<div class="tictactoe-grid">']
            for row_idx in range(self._board_size):
                row_number = self._board_size - row_idx
                for col_idx in range(self._board_size):
                    coord = self._coord_codec.index_to_coord(row_number - 1, col_idx)
                    cell_value = self._grid[row_idx][col_idx]
                    cell_html = self._render_cell(cell_value)

                    cell_class = "tictactoe-cell"
                    if self._last_move and coord == self._last_move:
                        cell_class += " last-move"
                    if coord in self._winning_line:
                        cell_class += " win-line"

                    coord_attr = html.escape(coord)
                    cell_attrs = [
                        f'class="{cell_class}"',
                        f'data-coord="{coord_attr}"',
                        f'aria-label="{coord_attr}"',
                        f'title="{coord_attr}"',
                    ]
                    if interactive:
                        coord_js = coord.replace("\\", "\\\\").replace("'", "\\'")
                        cell_attrs.extend(
                            [
                                'role="button"',
                                'tabindex="0"',
                                f'onclick="window.__gomoku_submit && window.__gomoku_submit(\'{coord_js}\')"',
                            ]
                        )
                    grid_blocks.append(f'<div {" ".join(cell_attrs)}>{cell_html}</div>')
            grid_blocks.append("</div>")
            return "\n".join(grid_blocks)

        blocks = [
            f'<div class="tictactoe-board-wrapper" data-interactive="{str(interactive).lower()}" {wrapper_style}>'
        ]
        blocks.append(render_col_labels_row())
        if self._show_coords:
            blocks.append('<div class="tictactoe-board-middle">')
            blocks.append('<div class="tictactoe-coords-col">')
            for row_idx in range(self._board_size):
                label = str(self._board_size - row_idx)
                blocks.append(f'<div class="tictactoe-coord-label">{label}</div>')
            blocks.append("</div>")
            blocks.append(render_grid())
            blocks.append('<div class="tictactoe-coords-col">')
            for row_idx in range(self._board_size):
                label = str(self._board_size - row_idx)
                blocks.append(f'<div class="tictactoe-coord-label">{label}</div>')
            blocks.append("</div>")
            blocks.append("</div>")
        else:
            blocks.append(render_grid())
        blocks.append(render_col_labels_row())
        blocks.append("</div>")
        return "\n".join(blocks)

    def _resolve_cell_size(self) -> int:
        if self._board_size <= 3:
            return 96
        if self._board_size == 4:
            return 80
        return max(48, int(320 / self._board_size))

    def _default_grid(self) -> list[list[str]]:
        return [[EMPTY_CELL for _ in range(self._board_size)] for _ in range(self._board_size)]

    def _column_labels(self) -> list[str]:
        return self._coord_codec.column_labels()

    def _parse_board_text(self, board_text: str) -> Optional[list[list[str]]]:
        lines = [line.strip() for line in board_text.splitlines() if line.strip()]
        if not lines:
            return None
        header_tokens = lines[0].split()
        has_header = bool(header_tokens) and all(token.isalpha() or token.isdigit() for token in header_tokens)
        start_idx = 1 if has_header else 0
        rows: list[list[str]] = []
        for line in lines[start_idx:]:
            tokens = [token for token in line.split() if token]
            if not tokens:
                continue
            if tokens[0].isdigit():
                tokens = tokens[1:]
            rows.append(tokens)
        if not rows:
            return None
        row_len = len(rows[0])
        if row_len == 0:
            return None
        if any(len(row) != row_len for row in rows):
            return None
        if has_header and len(header_tokens) != row_len:
            return None
        if len(rows) != row_len:
            return None
        return rows

    def _render_text(self, grid: list[list[str]]) -> str:
        labels = self._column_labels()
        lines = ["   " + " ".join(labels)]
        for row_idx in range(self._board_size - 1, -1, -1):
            row_label = f"{row_idx + 1:>2}"
            row = grid[row_idx] if row_idx < len(grid) else [EMPTY_CELL for _ in range(self._board_size)]
            lines.append(f"{row_label} " + " ".join(row))
        return "\n".join(lines)

    def _normalize_coord(self, coord: Optional[str]) -> Optional[str]:
        if not coord:
            return None
        value = str(coord).strip()
        if not value:
            return None
        if self._coord_scheme == "ROW_COL":
            match = re.match(r"^\s*(\d{1,3})\s*[,:\s]+\s*(\d{1,3})\s*$", value)
            if match:
                return f"{int(match.group(1))},{int(match.group(2))}"
        return value.upper()

    def _normalize_coords(self, coords: Optional[Sequence[str]]) -> set[str]:
        if not coords:
            return set()
        normalized = {value for value in (self._normalize_coord(coord) for coord in coords) if value}
        return normalized

    @staticmethod
    def _render_cell(cell: Any) -> str:
        value = str(cell or "").strip()
        if value in {"X", "x"}:
            return '<span class="tictactoe-mark x">X</span>'
        if value in {"O", "o"}:
            return '<span class="tictactoe-mark o">O</span>'
        if value in {EMPTY_CELL, "-", "_", ""}:
            return ""
        return f'<span class="tictactoe-mark">{html.escape(value)}</span>'


__all__ = ["TICTACTOE_BOARD_CSS", "TicTacToeBoardRenderer"]
