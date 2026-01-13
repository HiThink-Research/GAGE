"""Text-only renderer for Doudizhu arena snapshots."""

from __future__ import annotations

import html
from typing import Optional, Sequence

from gage_eval.registry import registry


@registry.asset(
    "renderer_impls",
    "doudizhu_text_v1",
    desc="Doudizhu text renderer for arena snapshots",
    tags=("doudizhu", "renderer", "card"),
)
class DoudizhuTextRenderer:
    """Render Doudizhu state snapshots as formatted text."""

    def __init__(self, board_size: int = 0, coord_scheme: str = "A1", **_: object) -> None:
        """Initialize the text renderer.

        Args:
            board_size: Unused placeholder for interface compatibility.
            coord_scheme: Unused placeholder for interface compatibility.
            **_: Ignored extra keyword arguments.
        """

        self._board_size = int(board_size)
        self._coord_scheme = str(coord_scheme)
        self._raw_text = ""
        self._last_move: Optional[str] = None

    def update(
        self,
        board_text: str,
        *,
        last_move: Optional[str] = None,
        winning_line: Optional[Sequence[str]] = None,
    ) -> None:
        """Update the renderer with a new text snapshot.

        Args:
            board_text: Raw text representation of the game state.
            last_move: Optional last move description.
            winning_line: Unused placeholder for interface compatibility.
        """

        _ = winning_line
        self._raw_text = board_text or ""
        self._last_move = last_move

    def resize(self, board_size: int) -> None:
        """Resize the renderer when requested by the caller."""

        self._board_size = int(board_size)

    def set_coord_scheme(self, coord_scheme: str) -> None:
        """Update the coordinate scheme metadata."""

        self._coord_scheme = str(coord_scheme)

    def render_html(self, *, interactive: bool) -> str:
        """Render the current snapshot into HTML markup.

        Args:
            interactive: Whether interactive mode is enabled.

        Returns:
            HTML snippet for the current state.
        """

        _ = interactive
        safe_text = html.escape(self._raw_text)
        last_move = f"<div class=\"doudizhu-last-move\">Last move: {html.escape(self._last_move)}</div>" if self._last_move else ""
        return (
            "<div class=\"doudizhu-shell\">"
            f"{last_move}"
            f"<pre class=\"doudizhu-board\">{safe_text}</pre>"
            "</div>"
        )

    def raw_text(self) -> str:
        """Return the latest raw board text."""

        return self._raw_text

    def get_css(self) -> str:
        """Return CSS required for the renderer."""

        return """
.doudizhu-shell {
  font-family: "IBM Plex Mono", "Menlo", "Monaco", monospace;
  color: #1f2933;
  background: #f8f4ea;
  border: 1px solid #e4d9c7;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 8px 18px rgba(0, 0, 0, 0.08);
}
.doudizhu-board {
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  margin: 0;
}
.doudizhu-last-move {
  font-weight: 700;
  margin-bottom: 8px;
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
        """Return JS for click-to-move interactions (unused for card games)."""

        _ = (
            board_container_id,
            move_input_id,
            submit_button_id,
            enable_click,
            refresh_button_id,
            refresh_interval_ms,
        )
        return ""


__all__ = ["DoudizhuTextRenderer"]
