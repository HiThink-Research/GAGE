"""Gradio-based visualizer for arena games."""

from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Optional, Sequence, Tuple

from loguru import logger

from gage_eval.registry import registry
from gage_eval.role.arena.visualizers.renderer_base import BoardRenderer

_BOARD_CONTAINER_ID = "gomoku-board-container"
_MOVE_INPUT_ID = "gomoku-move-input"
_SUBMIT_BUTTON_ID = "gomoku-move-submit"
_REFRESH_BUTTON_ID = "gomoku-refresh-button"
_FINISH_TIMEOUT_S = 15.0


class GradioVisualizer:
    """Embedded Gradio UI for observing or interacting with a game."""

    def __init__(
        self,
        *,
        board_size: int = 15,
        port: int = 7860,
        launch_browser: bool = False,
        mode: str = "observer",
        refresh_s: float = 0.3,
        auto_close: bool = False,
        wait_for_finish: bool = False,
        renderer_impl: str = "gomoku_board_v1",
        renderer_params: Optional[dict[str, object]] = None,
        coord_scheme: str = "A1",
        sanitize_output: bool = True,
        max_output_chars: int = 2000,
        show_parsed_move: bool = True,
        title: Optional[str] = None,
    ) -> None:
        """Initialize the visualizer settings.

        Args:
            board_size: Board dimension for the rendered grid.
            port: TCP port for the Gradio server.
            launch_browser: Whether to open a browser tab on launch.
            mode: UI mode ("observer" or "interactive").
            refresh_s: Refresh interval in seconds for polling updates.
        """

        self._board_size = int(board_size)
        self._port = int(port)
        self._launch_browser = bool(launch_browser)
        self._mode = mode
        self._refresh_s = float(refresh_s)
        self._auto_close = bool(auto_close)
        self._wait_for_finish = bool(wait_for_finish)
        self._action_queue: Queue[str] = Queue()
        self._lock = threading.Lock()
        self._board_text = ""
        self._renderer_impl = str(renderer_impl or "gomoku_board_v1")
        self._renderer_params = dict(renderer_params or {})
        self._coord_scheme = str(coord_scheme or "A1")
        self._renderer: BoardRenderer = self._build_renderer(self._board_size, self._coord_scheme)
        self._status_text = ""
        self._renderer_css = self._renderer.get_css() if hasattr(self._renderer, "get_css") else ""
        self._last_board_html = self._renderer.render_html(interactive=self._mode == "interactive")
        self._last_raw_text = self._renderer.raw_text()
        self._last_status = ""
        self._player_ids: list[str] = []
        self._player_names: dict[str, str] = {}
        self._player_labels: dict[str, str] = {}

        # Thinking timer state
        self._turn_start_time = time.time()
        self._current_active_player: Optional[str] = None

        self._player_bar_html = self._render_player_bar(active_player=self._current_active_player, elapsed=0.0)
        self._last_action_player: Optional[str] = None
        self._last_action_raw = ""
        self._last_action_move: Optional[str] = None
        self._last_output_label = self._render_output_label()
        self._finalized = False
        self._finish_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._demo = None
        self._sanitize_output = bool(sanitize_output)
        self._max_output_chars = max(0, int(max_output_chars))
        self._show_parsed_move = bool(show_parsed_move)
        self._title = str(title) if title else "GAGE Gomoku Arena"

    @property
    def action_queue(self) -> Queue[str]:
        """Return the action queue used for human input."""

        return self._action_queue

    def start(self) -> None:
        """Start the Gradio server in a background thread."""

        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._launch, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the Gradio server if possible."""

        self._running = False
        self._finalized = True
        self._finish_event.set()
        if self._demo is not None and self._auto_close:
            try:
                self._demo.close()
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("GradioVisualizer close failed: {}", exc)
            self._demo = None
        with self._lock:
            if not self._status_text:
                self._status_text = "Run finished"

    def _build_renderer(self, board_size: int, coord_scheme: str) -> BoardRenderer:
        try:
            renderer_cls = registry.get("renderer_impls", self._renderer_impl)
        except KeyError:
            registry.auto_discover("renderer_impls", "gage_eval.role.arena.games")
            renderer_cls = registry.get("renderer_impls", self._renderer_impl)
        params = dict(self._renderer_params)
        try:
            return renderer_cls(board_size=board_size, coord_scheme=coord_scheme, **params)
        except TypeError:
            return renderer_cls(board_size=board_size, **params)

    def update(
        self,
        *,
        board_text: str,
        status_text: str = "",
        last_move: Optional[str] = None,
        winning_line: Optional[Sequence[str]] = None,
        board_size: Optional[int] = None,
        last_action_player: Optional[str] = None,
        last_action_raw: Optional[str] = None,
        active_player: Optional[str] = None,
        coord_scheme: Optional[str] = None,
        last_action_move: Optional[str] = None,
        final_state: bool = False,
    ) -> None:
        """Update the rendered board and status text.

        Args:
            board_text: Text representation of the board.
            status_text: Optional status line to display.
            last_move: Optional last move coordinate for highlighting.
            board_size: Optional board size override from the environment.
            last_action_player: Optional player name for the last action.
            last_action_raw: Optional raw output for the last action.
            active_player: Optional player_id for the current turn.
            coord_scheme: Optional coordinate scheme for rendering.
            last_action_move: Optional parsed move coordinate for the last action.
            final_state: Whether this update represents a terminal state.
        """

        with self._lock:
            if board_size is not None:
                size = int(board_size)
                if size > 0 and size != self._board_size:
                    self._board_size = size
                    self._renderer.resize(size)
            if coord_scheme:
                self._coord_scheme = str(coord_scheme)
                if hasattr(self._renderer, "set_coord_scheme"):
                    self._renderer.set_coord_scheme(self._coord_scheme)
            self._board_text = board_text
            self._renderer.update(board_text, last_move=last_move, winning_line=winning_line)
            self._status_text = status_text
            if final_state and self._wait_for_finish:
                suffix = f"Click Finish to continue. Auto-confirm in {_FINISH_TIMEOUT_S:.0f}s."
                if suffix not in self._status_text:
                    self._status_text = f"{self._status_text} {suffix}".strip()
            self._last_board_html = self._renderer.render_html(
                interactive=self._mode == "interactive"
            )
            self._last_raw_text = self._renderer.raw_text()
            self._last_status = f'<div class="status-line">{self._status_text}</div>' if self._status_text else ""
            
            if last_action_player:
                self._last_action_player = last_action_player
            
            if last_action_raw is not None:
                raw_text = str(last_action_raw)
                if self._sanitize_output:
                    self._last_action_raw = self._sanitize_text(raw_text)
                else:
                    self._last_action_raw = raw_text

            if last_action_move is not None:
                self._last_action_move = str(last_action_move).strip() or None
            
            self._last_output_label = self._render_output_label()
            
            # Determine active player (thinking state).
            if final_state:
                self._current_active_player = None
            elif active_player:
                if active_player != self._current_active_player:
                    self._current_active_player = active_player
                    self._turn_start_time = time.time()
            elif self._last_action_player:
                next_player = self._next_player_id(self._last_action_player)
                if next_player and next_player != self._current_active_player:
                    self._current_active_player = next_player
                    self._turn_start_time = time.time()

            # No need to render bar here, _refresh will do it with live timer

            if final_state:
                self._finalized = True

    def _launch(self) -> None:
        try:
            import gradio as gr
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Gradio not available: {}", exc)
            self._running = False
            return

        refresh_interval_ms = max(100, int(self._refresh_s * 1000))
        js = ""
        if hasattr(self._renderer, "build_interaction_js"):
            js = self._renderer.build_interaction_js(
                board_container_id=_BOARD_CONTAINER_ID,
                move_input_id=_MOVE_INPUT_ID,
                submit_button_id=_SUBMIT_BUTTON_ID,
                enable_click=self._mode == "interactive",
                refresh_button_id=_REFRESH_BUTTON_ID,
                refresh_interval_ms=refresh_interval_ms,
            )

        launch_kwargs = {
            "server_port": self._port,
            "inbrowser": self._launch_browser,
            "share": False,
            "show_error": False,
        }
        blocks_kwargs = {"title": self._title}
        if self._renderer_css:
            blocks_kwargs["css"] = self._renderer_css
        if js:
            blocks_kwargs["js"] = js

        try:
            blocks = gr.Blocks(**blocks_kwargs)
        except TypeError as exc:  # pragma: no cover - compatibility fallback
            logger.warning("Gradio Blocks kwargs rejected: {}", exc)
            blocks = gr.Blocks(title=blocks_kwargs.get("title", "GAGE Gomoku Arena"))

        with blocks as demo:
            gr.Markdown(f"# {self._title}")
            with gr.Group(elem_id="gomoku-shell"):
                player_bar = gr.HTML(value=self._player_bar_html)
                with gr.Row(elem_id="gomoku-layout"):
                    with gr.Column(scale=2, elem_id="gomoku-left-panel"):
                        # Status Display
                        status_display = gr.HTML(value="")
                        
                        # Finish Button
                        finish_button = gr.Button(
                            "Finish", 
                            elem_id="gomoku-finish-button", 
                            visible=self._wait_for_finish,
                            interactive=False
                        )

                        # Board
                        board_display = gr.HTML(
                            value=self._renderer.render_html(interactive=self._mode == "interactive"),
                            elem_id=_BOARD_CONTAINER_ID,
                            sanitize_html=False,
                        )

                        if self._mode == "interactive":
                            with gr.Row(elem_id="gomoku-move-row"):
                                move_input = gr.Textbox(
                                    label="Your move",
                                    placeholder="e.g. H8",
                                    elem_id=_MOVE_INPUT_ID,
                                )
                                submit = gr.Button("Submit", elem_id=_SUBMIT_BUTTON_ID)

                            def _submit_move(text: str) -> Tuple[str, str]:
                                logger.info(f"[_submit_move] Triggered with text: '{text}'")
                                move = (text or "").strip().upper()
                                if move:
                                    self._action_queue.put(move)
                                    return "", self._last_status
                                return "", self._last_status

                            submit.click(_submit_move, inputs=[move_input], outputs=[move_input, status_display])

                        if self._wait_for_finish:
                            def _finish() -> str:
                                def _delayed_signal():
                                    time.sleep(0.5)
                                    self._finish_event.set()
                                threading.Thread(target=_delayed_signal, daemon=True).start()
                                return '<div class="status-line">Finish requested. Closing when pipeline resumes.</div>'

                            finish_button.click(_finish, outputs=[status_display])

                        refresh_button = gr.Button("Refresh", elem_id=_REFRESH_BUTTON_ID)

                        with gr.Accordion(label="Raw Board Text", open=True):
                            raw_display = gr.Textbox(
                                value="",
                                lines=max(4, self._board_size + 2),
                                interactive=False,
                            )
                    with gr.Column(scale=1, elem_id="gomoku-right-panel"):
                        output_label = gr.HTML(value=self._last_output_label, elem_id="gomoku-output-label")
                        output_box = gr.Textbox(
                            value="",
                            lines=50,
                            max_lines=5000, 
                            interactive=False,
                            show_label=False,
                            elem_id="gomoku-output-box",
                        )

            def _refresh(_: Optional[float] = None):
                finish_interactive = False
                finish_classes = []
                if self._finalized:
                    finish_interactive = True
                    finish_classes = ["finish-pulse"]
                
                finish_update = gr.update(
                    interactive=finish_interactive,
                    elem_classes=finish_classes
                )

                # Update timer
                elapsed = 0.0
                if self._current_active_player and not self._finalized:
                     elapsed = time.time() - self._turn_start_time
                
                # Regenerate player bar with new time
                self._player_bar_html = self._render_player_bar(active_player=self._current_active_player, elapsed=elapsed)

                if not self._running or self._finalized:
                    return (
                        gr.update(value=self._last_board_html),
                        self._status_text if "class" in self._status_text else self._last_status,
                        self._last_raw_text,
                        self._player_bar_html,
                        self._last_output_label,
                        self._last_action_raw,
                        finish_update,
                    )
                try:
                    board, status, raw_text = self._snapshot()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("GradioVisualizer refresh failed: {}", exc)
                    board, status, raw_text = self._last_board_html, self._last_status, self._last_raw_text
                
                return (
                    gr.update(value=board),
                    status,
                    raw_text,
                    self._player_bar_html,
                    self._last_output_label,
                    self._last_action_raw,
                    finish_update,
                )

            outputs_list = [
                board_display,
                status_display,
                raw_display,
                player_bar,
                output_label,
                output_box,
                finish_button,
            ]

            demo.load(
                _refresh,
                outputs=outputs_list,
            )
            refresh_button.click(
                _refresh,
                outputs=outputs_list,
            )
            if hasattr(gr, "Timer"):
                try:
                    timer = gr.Timer(value=self._refresh_s)
                    timer.tick(
                        _refresh,
                        outputs=outputs_list,
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning("Gradio Timer unavailable: {}", exc)

        demo.queue()
        self._demo = demo
        logger.info("Starting GradioVisualizer on port {}", self._port)
        try:
            demo.launch(**launch_kwargs)
        except TypeError as exc:  # pragma: no cover - compatibility fallback
            logger.warning("Gradio launch kwargs rejected: {}", exc)
            demo.launch()

    def _snapshot(self) -> Tuple[str, str, str]:
        try:
            with self._lock:
                raw_text = self._renderer.raw_text()
                board = self._renderer.render_html(interactive=self._mode == "interactive")
                status = f'<div class="status-line">{self._status_text}</div>' if self._status_text else ""
        except Exception:
            return self._last_board_html, self._last_status, self._last_raw_text
        if not board:
            board = self._renderer.render_html(interactive=self._mode == "interactive")
        if not raw_text:
            raw_text = self._board_text
        self._last_board_html = board
        self._last_raw_text = raw_text
        self._last_status = status
        return board, status, raw_text

    def wait_for_finish(self, timeout: Optional[float] = None) -> bool:
        """Block until the finish button is clicked when enabled."""

        if not self._wait_for_finish or not self._running:
            return True
        wait_timeout = _FINISH_TIMEOUT_S if timeout is None else max(0.0, float(timeout))
        completed = self._finish_event.wait(wait_timeout)
        if not completed:
            self._finish_event.set()
        return completed

    def reset_state(self) -> None:
        """Reset finish state for a new game session."""

        self._finalized = False
        self._finish_event.clear()
        with self._lock:
            self._status_text = ""
            self._last_status = ""
            self._last_action_player = None
            self._last_action_raw = ""
            self._last_action_move = None
            self._last_output_label = self._render_output_label()
            self._current_active_player = self._player_ids[0] if self._player_ids else None
            self._turn_start_time = time.time()
            self._player_bar_html = self._render_player_bar(
                active_player=self._current_active_player,
                elapsed=0.0,
            )

    def set_players(
        self,
        *,
        player_ids: Sequence[str],
        player_names: Optional[dict[str, str]] = None,
        player_labels: Optional[dict[str, str]] = None,
        active_player: Optional[str] = None,
    ) -> None:
        """Set player identity and labels for display."""

        ids = [str(player_id) for player_id in player_ids if player_id]
        if not ids:
            return
        names = dict(player_names or {})
        labels = dict(player_labels or {})
        with self._lock:
            self._player_ids = ids
            self._player_names = {player_id: str(names.get(player_id, player_id)) for player_id in ids}
            self._player_labels = {player_id: str(labels.get(player_id, player_id)) for player_id in ids}
            resolved_active = active_player if active_player in ids else self._current_active_player
            if resolved_active not in ids:
                resolved_active = ids[0]
            if resolved_active != self._current_active_player:
                self._current_active_player = resolved_active
                self._turn_start_time = time.time()
            elapsed = 0.0
            if self._current_active_player:
                elapsed = time.time() - self._turn_start_time
            self._player_bar_html = self._render_player_bar(
                active_player=self._current_active_player,
                elapsed=elapsed,
            )
            self._last_output_label = self._render_output_label()

    def _render_player_bar(self, active_player: Optional[str] = None, elapsed: float = 0.0) -> str:
        if not self._player_ids:
            return (
                '<div class="gomoku-players">'
                '<div class="gomoku-player neutral">'
                '<div class="player-label">Player</div>'
                '<div class="player-name">Waiting for players</div>'
                "</div>"
                "</div>"
            )

        # Format time (e.g., 0:05, 1:23)
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes}:{seconds:02d}"

        badge_html = (
            '<div class="status-badge">'
            '<span class="thinking-text">Thinking...</span>'
            f'<span class="timer-text">{time_str}</span>'
            "</div>"
        )

        blocks = ['<div class="gomoku-players">']
        for player_id in self._player_ids:
            tone = self._tone_for_player(player_id)
            classes = f"gomoku-player {tone}".strip()
            badge = ""
            if active_player == player_id:
                classes = f"{classes} thinking"
                badge = badge_html
            display_name = self._player_names.get(player_id, player_id)
            model_label = self._player_labels.get(player_id, player_id)
            blocks.append(
                f'<div class="{classes}">'
                f"{badge}"
                f'<div class="player-label">{display_name}</div>'
                f'<div class="player-name">{model_label}</div>'
                "</div>"
            )
        blocks.append("</div>")
        return "".join(blocks)

    def _render_output_label(self) -> str:
        if not self._last_action_player:
            return '<div class="gomoku-output-label">Awaiting first move</div>'
        player_id = self._last_action_player
        display_name = self._player_names.get(player_id, player_id)
        player_label = self._player_labels.get(player_id, player_id)
        tone = self._tone_for_player(player_id)
        move_html = ""
        if self._show_parsed_move and self._last_action_move:
            move_html = f'<span class="move-pill">{self._last_action_move}</span>'
        return (
            '<div class="gomoku-output-label">'
            f'<span class="player-pill {tone}">{display_name}</span>'
            f'<span class="player-name-inline">{player_label}</span>'
            f"{move_html}"
            "</div>"
        )

    def _sanitize_text(self, text: str) -> str:
        cleaned = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
        if self._max_output_chars <= 0 or len(cleaned) <= self._max_output_chars:
            return cleaned
        suffix = "\n... (truncated)"
        return f"{cleaned[: self._max_output_chars]}{suffix}"

    def _tone_for_player(self, player_id: str) -> str:
        if not self._player_ids:
            return "neutral"
        if player_id == self._player_ids[0]:
            return "black"
        if len(self._player_ids) > 1 and player_id == self._player_ids[1]:
            return "white"
        return "neutral"

    def _next_player_id(self, player_id: str) -> Optional[str]:
        if not self._player_ids or player_id not in self._player_ids:
            return None
        idx = self._player_ids.index(player_id)
        return self._player_ids[(idx + 1) % len(self._player_ids)]
