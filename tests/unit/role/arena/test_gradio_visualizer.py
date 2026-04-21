from __future__ import annotations

from typing import Optional, Sequence

import pytest

from gage_eval.role.arena.visualizers.gradio_visualizer import GradioVisualizer


class _DummyRenderer:
    def __init__(self) -> None:
        self._board_text = ""

    def update(
        self,
        board_text: str,
        *,
        last_move: Optional[str] = None,
        winning_line: Optional[Sequence[str]] = None,
    ) -> None:
        _ = last_move, winning_line
        self._board_text = board_text

    def render_html(self, *, interactive: bool = False) -> str:
        _ = interactive
        return f"<div>{self._board_text}</div>"

    def raw_text(self) -> str:
        return self._board_text

    def resize(self, board_size: int) -> None:
        _ = board_size

    def set_coord_scheme(self, coord_scheme: str) -> None:
        _ = coord_scheme

    def get_css(self) -> str:
        return ""


@pytest.fixture
def patch_renderer(monkeypatch):
    monkeypatch.setattr(
        GradioVisualizer,
        "_build_renderer",
        lambda self, board_size, coord_scheme: _DummyRenderer(),
    )


def test_gradio_visualizer_escapes_status_text_by_default(patch_renderer) -> None:
    visualizer = GradioVisualizer()

    visualizer.update(board_text="board", status_text="<b>unsafe</b>")

    assert "&lt;b&gt;unsafe&lt;/b&gt;" in visualizer._last_status
    assert '<div class="status-line"><b>unsafe</b></div>' not in visualizer._last_status


def test_gradio_visualizer_allows_status_html_when_explicitly_enabled(patch_renderer) -> None:
    visualizer = GradioVisualizer(allow_status_html=True)

    visualizer.update(board_text="board", status_text="<b>trusted</b>")

    assert '<div class="status-line"><b>trusted</b></div>' == visualizer._last_status


def test_gradio_visualizer_keeps_errors_visible_by_default(patch_renderer) -> None:
    visualizer = GradioVisualizer()

    assert visualizer._build_launch_kwargs()["show_error"] is True  # noqa: SLF001
    assert visualizer._build_error_suppression_js() == ""  # noqa: SLF001


def test_gradio_visualizer_demo_mode_can_suppress_frontend_errors(patch_renderer) -> None:
    visualizer = GradioVisualizer(demo_mode=True)

    assert visualizer._build_launch_kwargs()["show_error"] is False  # noqa: SLF001
    assert ".gradio-error" in visualizer._build_error_suppression_js()  # noqa: SLF001


def test_gradio_visualizer_bounds_output_history_and_keeps_monotonic_sequence(
    patch_renderer,
) -> None:
    visualizer = GradioVisualizer(max_output_entries=2)

    visualizer._record_output_history(player_id="p0", move="A1", raw_text="first")  # noqa: SLF001
    visualizer._record_output_history(player_id="p0", move="B2", raw_text="second")  # noqa: SLF001
    visualizer._record_output_history(player_id="p0", move="C3", raw_text="third")  # noqa: SLF001

    assert len(visualizer._output_history) == 2  # noqa: SLF001
    assert "[001]" not in visualizer._output_history_text  # noqa: SLF001
    assert "[002]" in visualizer._output_history_text  # noqa: SLF001
    assert "[003]" in visualizer._output_history_text  # noqa: SLF001
