"""Gomoku board renderer for HTML-based visualization."""

from __future__ import annotations

import html
from typing import Any, Optional, Sequence

from gage_eval.registry import registry
from gage_eval.role.arena.games.gomoku.coord_scheme import GomokuCoordCodec, normalize_coord_scheme
from gage_eval.role.arena.visualizers.utils import build_board_interaction_js

GOMOKU_BOARD_CSS = """
.gomoku-shell {
  font-family: 'Source Sans Pro', 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
  color: #333;
  -webkit-font-smoothing: antialiased;
  max-width: 100%;
  margin: 0 auto;
}

#gomoku-layout {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: stretch;
}

#gomoku-left-panel,
#gomoku-right-panel {
  background: #ffffff;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  border: 1px solid #e0e0e0;
}

#gomoku-left-panel {
  flex: 2;
  min-width: 360px;
  display: flex;
  flex-direction: column;
  align-items: center;
}

#gomoku-right-panel {
  flex: 1;
  min-width: 300px;
  display: flex;
  flex-direction: column;
}

/* Players */
.gomoku-players {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
  width: 100%;
}
.gomoku-player {
  flex: 1;
  padding: 16px;
  border-radius: 12px;
  background: #fff;
  border: 1px solid #eaeaea;
  box-shadow: 0 2px 8px rgba(0,0,0,0.02);
  display: flex;
  flex-direction: column;
  position: relative;
  overflow: hidden;
}
.gomoku-player::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 6px;
}
.gomoku-player.black {
  background: #1a1a1a;
  border: 1px solid #000;
  color: #fff;
}
.gomoku-player.black::before { background: #444; }
.gomoku-player.black .player-label { color: #888; }
.gomoku-player.black .player-name { color: #fff; }

.gomoku-player.white {
  background: #fff;
}
.gomoku-player.white::before { background: #ccc; }

.player-label {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #999;
  margin-bottom: 6px;
  padding-left: 12px;
}
.player-name {
  font-size: 24px;
  font-weight: 600;
  color: #222;
  padding-left: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.2;
}

/* Board Container */
#gomoku-board-container {
  display: flex;
  justify-content: center;
  margin: 0;
  perspective: 1000px;
}

.gomoku-board-wrapper {
  background-color: #eebb66;
  background-image: 
    linear-gradient(rgba(255, 255, 255, 0.1) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.1) 1px, transparent 1px),
    url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.6' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.08'/%3E%3C/svg%3E");
  padding: 24px;
  border-radius: 8px;
  box-shadow: 
    0 10px 40px rgba(0,0,0,0.4),
    inset 0 0 0 1px rgba(255,255,255,0.2),
    inset 0 0 30px rgba(0,0,0,0.1);
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  border: 1px solid #cfaa5e;
}

/* Coordinate System - Precise Alignment */
.gomoku-coords-row {
  display: flex;
  height: 24px;
}
.gomoku-coords-row.top { margin-bottom: 2px; }
.gomoku-coords-row.bottom { margin-top: 2px; }

.corner-spacer {
  width: 24px;
  height: 24px;
  flex-shrink: 0;
}
.gomoku-coords-row .corner-spacer:first-child { margin-right: 2px; }
.gomoku-coords-row .corner-spacer:last-child { margin-left: 2px; }

.gomoku-coords-col {
  display: flex;
  flex-direction: column;
  width: 24px;
  padding-top: 2px;
  padding-bottom: 2px;
}
.gomoku-coords-col.left { margin-right: 2px; }
.gomoku-coords-col.right { margin-left: 2px; }

.coord-label {
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'IBM Plex Mono', 'JetBrains Mono', monospace;
  font-size: 11px;
  color: #5e4022;
  opacity: 0.9;
  font-weight: 600;
  flex-shrink: 0;
}

.gomoku-board-middle {
  display: flex;
  align-items: flex-start;
}

/* Grid */
.gomoku-grid {
  display: flex;
  flex-direction: column;
  background: rgba(0,0,0,0.02);
  border: 2px solid #5e4022;
  box-sizing: content-box;
}

.gomoku-row {
  display: flex;
}
.gomoku-cell {
  width: 36px;
  height: 36px;
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: pointer;
}
.gomoku-cell::before { content: ""; position: absolute; top: 50%; left: 0; width: 100%; height: 1px; background: #5e4022; transform: translateY(-50%); z-index: 0; }
.gomoku-cell::after { content: ""; position: absolute; top: 0; left: 50%; height: 100%; width: 1px; background: #5e4022; transform: translateX(-50%); z-index: 0; }
.gomoku-board-wrapper[data-interactive="true"] .gomoku-cell:not(:has(.piece)):hover::before,
.gomoku-board-wrapper[data-interactive="true"] .gomoku-cell:not(:has(.piece)):hover::after { background: #a37c54; box-shadow: 0 0 4px rgba(0,0,0,0.2); }
.gomoku-cell.star-point .star-marker { position: absolute; width: 5px; height: 5px; background-color: #332211; border-radius: 50%; z-index: 1; top: 50%; left: 50%; transform: translate(-50%, -50%); box-shadow: 0 1px 1px rgba(255,255,255,0.2); }
.piece { width: 31px; height: 31px; border-radius: 50%; z-index: 10; position: relative; box-shadow: 2px 3px 6px rgba(0, 0, 0, 0.4); transform: scale(0.95); animation: drop-in 0.25s cubic-bezier(0.18, 0.89, 0.32, 1.28); }
@keyframes drop-in { 0% { transform: scale(1.5); opacity: 0; } 100% { transform: scale(0.95); opacity: 1; } }
.piece.black { background: radial-gradient(circle at 35% 35%, #555, #111 50%, #000 100%); }
.piece.white { background: radial-gradient(circle at 35% 35%, #fff, #f0f0f0 40%, #ddd 100%); }
.gomoku-cell.last-move .piece::after { content: ""; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 6px; height: 6px; background-color: #f44336; border-radius: 50%; box-shadow: 0 0 4px #f44336; }
.gomoku-cell.win-line .piece { box-shadow: 0 0 0 2px #ffb300, 0 0 15px #ffb300; }

/* Output & Status */
#gomoku-output-box textarea {
  font-family: 'IBM Plex Mono', 'JetBrains Mono', monospace;
  font-size: 13px;
  line-height: 1.6;
  color: #444;
  background: #fafafa;
  border: 1px solid #eee;
  border-radius: 8px;
  padding: 16px;
  min-height: 1000px !important;
  height: 100% !important;
}

.gomoku-output-label {
  display: flex;
  align-items: center;
  gap: 12px;
  justify-content: flex-start;
  margin-bottom: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid #f0f0f0;
}
.player-pill {
  padding: 4px 12px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  flex-shrink: 0;
}
.player-pill.black { background: #333; color: #fff; }
.player-pill.white { background: #eee; color: #333; border: 1px solid #ddd; }
.player-name-inline {
  font-size: 15px;
  font-weight: 600;
  color: #222;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.move-pill {
  padding: 4px 10px;
  border-radius: 999px;
  background: #f2f2f2;
  font-size: 12px;
  font-weight: 700;
  color: #444;
  border: 1px solid #e0e0e0;
  letter-spacing: 0.5px;
}

.status-line {
    font-size: 16px;
    font-weight: 500;
    color: #444;
    padding: 12px 16px;
    background: #f8f9fa;
    border-radius: 8px;
    border-left: 4px solid #333;
    margin-bottom: 16px;
    display: block;
    width: 100%;
    box-sizing: border-box;
}

/* Move Input */
#gomoku-move-row {
  width: 100%;
  gap: 12px;
  align-items: center;
}
#gomoku-move-input {
  flex: 1 1 auto;
}
#gomoku-move-input textarea,
#gomoku-move-input input {
  min-height: 42px;
  font-size: 14px;
}
#gomoku-move-submit {
  flex: 0 0 auto;
}
#gomoku-move-submit button {
  min-height: 42px;
  min-width: 120px;
  font-weight: 600;
}

/* Finish Button */
#gomoku-finish-button {
  background: #555;
  color: white !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  margin-top: 10px;
  width: 100%;
}
.finish-pulse {
  animation: finish-pulse 2s infinite;
  background-color: #d32f2f !important;
}
@keyframes finish-pulse {
  0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(229, 57, 53, 0.7); }
  70% { transform: scale(1.02); box-shadow: 0 0 0 10px rgba(229, 57, 53, 0); }
  100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(229, 57, 53, 0); }
}

/* Thinking State Animation */
@keyframes thinking-pulse-black {
  0% { box-shadow: 0 2px 8px rgba(0,0,0,0.02); border-color: #000; }
  50% { box-shadow: 0 4px 20px rgba(0,0,0,0.5); border-color: #666; transform: translateY(-2px); }
  100% { box-shadow: 0 2px 8px rgba(0,0,0,0.02); border-color: #000; }
}
@keyframes thinking-pulse-white {
  0% { box-shadow: 0 2px 8px rgba(0,0,0,0.02); border-color: #eaeaea; }
  50% { box-shadow: 0 4px 20px rgba(0,0,0,0.15); border-color: #999; transform: translateY(-2px); }
  100% { box-shadow: 0 2px 8px rgba(0,0,0,0.02); border-color: #eaeaea; }
}

.gomoku-player.black.thinking {
  animation: thinking-pulse-black 2s infinite ease-in-out;
  border: 1px solid #444;
}
.gomoku-player.white.thinking {
  animation: thinking-pulse-white 2s infinite ease-in-out;
  border: 1px solid #bbb;
}

.status-badge {
  display: none;
  position: absolute;
  top: 16px;
  right: 16px;
  background: rgba(255, 255, 255, 0.95);
  padding: 4px 10px;
  border-radius: 6px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  align-items: center;
  gap: 8px;
  animation: fade-in 0.3s;
  border: 1px solid rgba(0,0,0,0.05);
  z-index: 20;
}
.gomoku-player.thinking .status-badge {
  display: flex;
}
.thinking-text {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  color: #888;
  letter-spacing: 0.5px;
}
.timer-text {
  font-family: 'IBM Plex Mono', 'JetBrains Mono', monospace;
  font-size: 14px;
  font-weight: 600;
  color: #d32f2f;
  min-width: 30px;
  text-align: right;
}

@keyframes fade-in { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }

.toast-wrap, .toast, .toast-container { display: none !important; }
"""


@registry.asset(
    "renderer_impls",
    "gomoku_board_v1",
    desc="Gomoku board renderer (HTML/CSS grid)",
    tags=("gomoku", "renderer"),
)
class GomokuBoardRenderer:
    """Render and parse grid-style boards from text snapshots."""

    def __init__(self, board_size: int, coord_scheme: str = "A1", show_coords: bool = True) -> None:
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
        """Return CSS for the Gomoku board."""
        
        return GOMOKU_BOARD_CSS + """
/* === Global Fixes === */
/* Hide Gradio error overlays/toasts */
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
/* Prevent layout jitter */
.gradio-container { overflow-y: scroll !important; }

/* === Layout Containers === */
#gomoku-layout {
  align-items: flex-start;
  gap: 24px;
}
#gomoku-left-panel {
  display: flex;
  flex-direction: column;
  align-items: center;
}

/* === Player Cards (Top Bar) === */
.gomoku-players {
  display: flex;
  gap: 20px;
  margin-bottom: 24px;
  width: 100%;
}

.gomoku-player {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 16px 20px;
  border-radius: 12px;
  background: white;
  border: 1px solid #eaeaea;
  position: relative;
  transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
  box-shadow: 0 4px 6px rgba(0,0,0,0.02);
}

/* Neutral / Waiting State */
.gomoku-player.neutral {
  background: #f9f9f9;
  border: 1px dashed #ccc;
  opacity: 0.7;
}

/* Black Player Theme */
.gomoku-player.black {
  background: linear-gradient(145deg, #2b2b2b, #000000);
  color: white;
  border: 1px solid #000;
}
.gomoku-player.black .player-label { color: rgba(255,255,255,0.6); }
.gomoku-player.black .player-name { color: #fff; }
.gomoku-player.black .timer-text { color: #ddd; }

/* White Player Theme */
.gomoku-player.white {
  background: linear-gradient(145deg, #ffffff, #f0f0f0);
  color: #333;
  border: 1px solid #ccc;
}
.gomoku-player.white .player-label { color: #666; }
.gomoku-player.white .player-name { color: #222; }

/* Typography */
.player-label {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 6px;
}
.player-name {
  font-size: 20px; /* Larger font */
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.2;
}

/* Active "Thinking" State */
.gomoku-player.thinking {
  z-index: 10;
  box-shadow: 0 10px 22px rgba(0,0,0,0.12);
}
.gomoku-player.thinking::after {
  content: "";
  position: absolute;
  left: 14px;
  right: 14px;
  bottom: 12px;
  height: 3px;
  border-radius: 999px;
  background-size: 220% 100%;
  animation: thinking-scan 1.6s linear infinite;
  opacity: 0.9;
}
.gomoku-player.black.thinking::after {
  background-image: linear-gradient(
    90deg,
    rgba(255,255,255,0.08),
    rgba(255,255,255,0.65),
    rgba(255,255,255,0.08)
  );
}
.gomoku-player.white.thinking::after {
  background-image: linear-gradient(
    90deg,
    rgba(0,0,0,0.08),
    rgba(0,0,0,0.35),
    rgba(0,0,0,0.08)
  );
}

@keyframes thinking-scan {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* Status Badge (Thinking... Timer) */
.status-badge {
  display: none;
  position: absolute;
  top: 16px;
  right: 16px;
}
.gomoku-player.thinking .status-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  background: rgba(255,255,255,0.9);
  padding: 4px 10px;
  border-radius: 999px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.gomoku-player.black .status-badge { background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.2); }
.gomoku-player.white .status-badge { background: rgba(255,255,255,0.9); border: 1px solid #ddd; }

.thinking-text {
  font-size: 10px; 
  font-weight: 800; 
  text-transform: uppercase; 
  letter-spacing: 0.5px;
}
.timer-text { font-family: monospace; font-weight: 600; font-size: 12px; }


/* === Board Area === */
.gomoku-board-wrapper {
  background-color: #eebb66;
  background-image: 
    linear-gradient(rgba(255, 255, 255, 0.1) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.1) 1px, transparent 1px),
    url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.6' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.08'/%3E%3C/svg%3E");
  padding: 24px;
  border-radius: 4px; /* Sharper corners for board feel */
  /* Deep shadow to lift off the gray background */
  box-shadow: 
    0 20px 50px rgba(0,0,0,0.3),
    0 0 0 1px #cfaa5e inset;
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  border: 4px solid #b58863; /* Frame */
  min-height: 540px;
  z-index: 5;
}

/* Coordinate Labels */
.coord-label {
  width: 36px; height: 36px;
  display: flex; align-items: center; justify-content: center;
  font-family: monospace; font-size: 11px; font-weight: 600;
  color: #5e4022; opacity: 0.9;
}
.gomoku-coords-row { display: flex; height: 24px; }
.corner-spacer { width: 24px; height: 24px; flex-shrink: 0; }
.gomoku-coords-col { display: flex; flex-direction: column; width: 24px; }

/* Grid & Cells */
.gomoku-grid {
  display: flex; flex-direction: column;
  background: rgba(0,0,0,0.02);
  border: 2px solid #5e4022;
  box-sizing: content-box;
}
.gomoku-row { display: flex; }
.gomoku-cell {
  width: 36px; height: 36px;
  position: relative;
  display: flex; justify-content: center; align-items: center;
  cursor: pointer;
}
/* Grid Lines */
.gomoku-cell::before { content: ""; position: absolute; top: 50%; left: 0; width: 100%; height: 1px; background: #5e4022; transform: translateY(-50%); z-index: 0; }
.gomoku-cell::after { content: ""; position: absolute; top: 0; left: 50%; height: 100%; width: 1px; background: #5e4022; transform: translateX(-50%); z-index: 0; }
/* Star Points */
.gomoku-cell.star-point .star-marker { 
  position: absolute; width: 6px; height: 6px; 
  background-color: #332211; border-radius: 50%; z-index: 1; 
}

/* Hover Effect */
.gomoku-board-wrapper[data-interactive="true"] .gomoku-cell:not(:has(.piece)):hover::before,
.gomoku-board-wrapper[data-interactive="true"] .gomoku-cell:not(:has(.piece)):hover::after { 
  background: #886644; box-shadow: 0 0 4px rgba(0,0,0,0.3); 
}

/* Pieces */
.piece {
  width: 30px; height: 30px;
  border-radius: 50%; z-index: 10;
  position: relative;
  box-shadow: 2px 3px 5px rgba(0,0,0,0.5);
  transform: scale(0.95);
  animation: drop-in 0.2s cubic-bezier(0.18, 0.89, 0.32, 1.28);
}
@keyframes drop-in { 0% { transform: scale(1.4); opacity: 0; } 100% { transform: scale(0.95); opacity: 1; } }

.piece.black { background: radial-gradient(circle at 35% 35%, #666, #000); }
.piece.white { background: radial-gradient(circle at 35% 35%, #fff, #ddd); }

/* Last Move & Win Line */
.gomoku-cell.last-move .piece::after {
  content: ""; position: absolute;
  top: 50%; left: 50%; transform: translate(-50%, -50%);
  width: 6px; height: 6px; background-color: #ff3333;
  border-radius: 50%; box-shadow: 0 0 4px #ff0000;
}
.gomoku-cell.win-line .piece {
  box-shadow: 0 0 0 3px #ffb300, 0 0 20px #ffb300;
}


/* === Right Panel (Log) === */
.gomoku-output-label {
  display: flex; align-items: center; gap: 10px;
  padding: 10px; border-bottom: 1px solid #eee;
  background: #fff; border-radius: 8px 8px 0 0;
}
#gomoku-output-box textarea {
  font-family: 'SF Mono', 'Menlo', 'Monaco', 'Courier New', monospace;
  font-size: 13px;
  line-height: 1.5;
  color: #333;
  background: #fcfcfc;
  border: 1px solid #e0e0e0;
  border-radius: 0 0 8px 8px; /* Connected to label */
  padding: 12px;
  box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
  min-height: 600px !important;
}

/* Pills */
.player-pill {
  padding: 4px 8px; border-radius: 4px;
  font-size: 11px; font-weight: 700; text-transform: uppercase;
}
.player-pill.black { background: #222; color: #fff; }
.player-pill.white { background: #eee; color: #333; border: 1px solid #ccc; }
.move-pill {
  background: #e3f2fd; color: #1565c0;
  padding: 2px 8px; border-radius: 12px;
  font-size: 12px; font-weight: 600;
}

/* === Hidden Controls === */
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
  width: 100%; margin-top: 12px;
}
#gomoku-refresh-button {
  display: none !important;
}
"""

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

        star_points = set()
        if self._board_size == 15:
            points = [(3, 3), (11, 3), (7, 7), (3, 11), (11, 11)]
            for row_idx, col_idx in points:
                star_points.add((row_idx, col_idx))

        def render_col_labels_row() -> str:
            html_parts = ['<div class="gomoku-coords-row">', '<div class="corner-spacer"></div>']
            for label in col_labels:
                html_parts.append(f'<div class="coord-label">{label}</div>')
            html_parts.append('<div class="corner-spacer"></div>')
            html_parts.append('</div>')
            return "".join(html_parts)

        def render_grid() -> str:
            grid_blocks = ['<div class="gomoku-grid">']
            for row_idx in range(self._board_size):
                row_number = self._board_size - row_idx
                grid_blocks.append('<div class="gomoku-row">')
                for col_idx in range(self._board_size):
                    coord = self._coord_codec.index_to_coord(row_number - 1, col_idx)
                    cell_value = self._grid[row_idx][col_idx]
                    cell_html = self._render_cell(cell_value)

                    is_star_point = (row_idx, col_idx) in star_points
                    if is_star_point and not cell_html:
                        cell_html = '<div class="star-marker"></div>'
                    elif is_star_point and cell_html:
                        cell_html = '<div class="star-marker"></div>' + cell_html

                    cell_class = "gomoku-cell"
                    if is_star_point:
                        cell_class += " star-point"
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
            grid_blocks.append("</div>")
            return "\n".join(grid_blocks)

        blocks = [f'<div class="gomoku-board-wrapper" data-interactive="{str(interactive).lower()}">']
        grid_html = render_grid()

        if self._show_coords:
            blocks.append(render_col_labels_row())
            blocks.append('<div class="gomoku-board-middle">')
            blocks.append('<div class="gomoku-coords-col left">')
            for row_idx in range(self._board_size):
                label = str(self._board_size - row_idx)
                blocks.append(f'<div class="coord-label">{label}</div>')
            blocks.append("</div>")
            blocks.append(grid_html)
            blocks.append('<div class="gomoku-coords-col right">')
            for row_idx in range(self._board_size):
                label = str(self._board_size - row_idx)
                blocks.append(f'<div class="coord-label">{label}</div>')
            blocks.append("</div>")
            blocks.append("</div>")
            blocks.append(render_col_labels_row())
        else:
            blocks.append(grid_html)

        blocks.append("</div>")
        return "\n".join(blocks)

    def _default_grid(self) -> list[list[str]]:
        return [["." for _ in range(self._board_size)] for _ in range(self._board_size)]

    def _column_labels(self) -> list[str]:
        return self._coord_codec.column_labels()

    def _parse_board_text(self, board_text: str) -> Optional[list[list[str]]]:
        lines = [line.strip() for line in board_text.splitlines() if line.strip()]
        if not lines:
            return None
        header_tokens = lines[0].split()
        has_header = bool(header_tokens) and all(
            token.isalpha() or token.isdigit() for token in header_tokens
        )
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
            row = grid[row_idx] if row_idx < len(grid) else [".." for _ in range(self._board_size)]
            lines.append(f"{row_label} " + " ".join(row))
        return "\n".join(lines)

    @staticmethod
    def _normalize_coord(coord: Optional[str]) -> Optional[str]:
        if not coord:
            return None
        value = str(coord).strip().upper()
        if not value:
            return None
        return value

    @classmethod
    def _normalize_coords(cls, coords: Optional[Sequence[str]]) -> set[str]:
        if not coords:
            return set()
        normalized = {value for value in (cls._normalize_coord(coord) for coord in coords) if value}
        return normalized

    @staticmethod
    def _render_cell(cell: Any) -> str:
        value = str(cell or "").strip()
        if value in {"B", "b", "X", "x", "●", "⚫"}:
            return '<div class="piece black"></div>'
        if value in {"W", "w", "O", "o", "○", "⚪"}:
            return '<div class="piece white"></div>'
        if value in {".", "-", "_", ""}:
            return ""
        return f'<span class="piece-text">{html.escape(value)}</span>'


__all__ = ["GOMOKU_BOARD_CSS", "GomokuBoardRenderer"]
