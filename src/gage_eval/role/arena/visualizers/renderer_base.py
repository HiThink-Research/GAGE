"""Renderer protocols for arena visualizers."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence


class BoardRenderer(Protocol):
    """Defines the minimal interface for board renderers."""

    def update(
        self,
        board_text: str,
        *,
        last_move: Optional[str] = None,
        winning_line: Optional[Sequence[str]] = None,
    ) -> None:
        """Update the renderer with a new board snapshot."""

    def resize(self, board_size: int) -> None:
        """Resize the renderer grid if needed."""

    def set_coord_scheme(self, coord_scheme: str) -> None:
        """Update the renderer coordinate scheme."""

    def render_html(self, *, interactive: bool) -> str:
        """Render the current grid into HTML markup."""

    def raw_text(self) -> str:
        """Return the latest raw board text."""

    def get_css(self) -> str:
        """Return CSS required for the renderer."""

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

