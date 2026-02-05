"""WebSocket server attached to an in-process stable-retro environment.

The standalone WebSocket server (`websocket_server.py`) owns its own environment
instance. This module provides an "attached" variant that:
- Streams frames from an existing `StableRetroArenaEnvironment` owned by the
  arena pipeline.
- Translates browser key events into macro moves and enqueues them into a queue
  consumed by `HumanAdapter` / `HumanPlayer`.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Any, Callable, Mapping, Optional, Sequence

from loguru import logger

from gage_eval.role.arena.games.retro.keyboard_input import KeyState, build_default_key_map
from gage_eval.role.arena.games.retro.websocket_tracks import encode_jpeg, parse_key_payload

try:
    import uvicorn
except Exception:  # pragma: no cover - optional import for runtime
    uvicorn = None  # type: ignore[assignment]

try:
    from fastapi import WebSocket as FastAPIWebSocket
    from fastapi import WebSocketDisconnect as FastAPIWebSocketDisconnect
except Exception:  # pragma: no cover - optional import for runtime
    FastAPIWebSocket = Any  # type: ignore[misc,assignment]
    FastAPIWebSocketDisconnect = Exception  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class AttachedServerConfig:
    """Configuration for an attached retro WebSocket server."""

    host: str = "0.0.0.0"
    port: int = 5800
    fps: int = 30
    frame_queue_size: int = 2
    display_mode: str = "headless"
    legal_moves: Optional[Sequence[str]] = None
    hold_ticks_default: int = 6


def _load_web_deps() -> tuple[Any, Any, Any, Any]:
    try:
        from fastapi import FastAPI
        from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
    except ImportError as exc:
        raise ImportError(
            "Attached WebSocket server dependencies missing. Install fastapi + uvicorn."
        ) from exc
    if uvicorn is None:
        raise ImportError("Attached WebSocket server requires uvicorn to run.")
    return FastAPI, FileResponse, HTMLResponse, PlainTextResponse


def _enqueue_move(action_queue: Queue[str], *, move: str, hold_ticks: int) -> None:
    payload = {"move": str(move), "hold_ticks": int(hold_ticks)}
    action_queue.put(json.dumps(payload, ensure_ascii=False))


class RetroAttachedWebSocketServer:
    """Serve a WebSocket UI that controls an existing retro environment."""

    def __init__(
        self,
        *,
        config: AttachedServerConfig,
        frame_source: Callable[[], Optional[Any]],
        action_queue: Optional[Queue[str]],
    ) -> None:
        """Initialize the attached server.

        Args:
            config: Server configuration.
            frame_source: Callable that returns the latest RGB frame (numpy array).
            action_queue: Optional queue that receives JSON action strings. When
                omitted, the server streams frames but ignores keyboard input.
        """

        self._config = config
        self._frame_source = frame_source
        self._action_queue = action_queue
        self._key_state = KeyState(build_default_key_map(), legal_moves=config.legal_moves)
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[Any] = None

    @property
    def action_queue(self) -> Optional[Queue[str]]:
        return self._action_queue

    def start(self) -> None:
        """Start the attached server in a background thread."""

        if self._thread and self._thread.is_alive():
            return

        def _runner() -> None:
            try:
                self._run()
            except Exception as exc:  # pragma: no cover - runtime-only failure path
                logger.exception("Retro attached WebSocket server failed: {}", exc)

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()
        logger.info("Retro attached WebSocket server listening on http://localhost:{}/", self._config.port)

    def stop(self) -> None:
        """Stop the server."""

        server = self._server
        if server is None:
            return
        try:
            server.should_exit = True
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        FastAPI, FileResponse, HTMLResponse, PlainTextResponse = _load_web_deps()
        from pathlib import Path

        client_dir = Path(__file__).with_name("websocket_client")
        index_path = client_dir / "index.html"
        client_js_path = client_dir / "client.js"
        logger.info(
            "Starting retro attached WS server host={} port={} fps={} client_dir={}",
            self._config.host,
            self._config.port,
            self._config.fps,
            client_dir,
        )

        app = FastAPI()
        frame_queue: "asyncio.Queue[Optional[Any]]" = asyncio.Queue(maxsize=self._config.frame_queue_size)

        @app.on_event("startup")
        async def _startup() -> None:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._frame_publisher(loop, frame_queue))
            logger.info("Retro attached WebSocket server started (fps={})", self._config.fps)
            logger.info("Keyboard mapping (key -> action): {}", self._key_state.key_map)

        @app.get("/")
        async def index() -> Any:
            if not index_path.exists():
                return PlainTextResponse("Missing client bundle", status_code=404)
            return HTMLResponse(index_path.read_text(encoding="utf-8"))

        @app.get("/client.js")
        async def client_js() -> Any:
            if not client_js_path.exists():
                return PlainTextResponse("Missing client bundle", status_code=404)
            return FileResponse(client_js_path)

        @app.websocket("/ws")
        async def ws_endpoint(ws: FastAPIWebSocket) -> None:
            """Stream JPEG frames and accept keyboard input over WebSocket."""

            await ws.accept()
            last_enqueued_move: Optional[str] = None

            async def _recv_loop() -> None:
                nonlocal last_enqueued_move
                while True:
                    message = await ws.receive_text()
                    payload = parse_key_payload(message)
                    if not payload:
                        continue
                    _apply_ws_input(payload, self._key_state)
                    move = self._key_state.resolve_move()
                    if self._action_queue is None:
                        last_enqueued_move = move
                        continue
                    if move != last_enqueued_move:
                        last_enqueued_move = move
                        _enqueue_move(
                            self._action_queue,
                            move=move,
                            hold_ticks=self._config.hold_ticks_default,
                        )

            recv_task = asyncio.create_task(_recv_loop())
            try:
                while True:
                    frame = await frame_queue.get()
                    if frame is None:
                        continue
                    jpeg_bytes, error = encode_jpeg(frame, quality=80)
                    if error:
                        await ws.send_text(json.dumps({"ok": False, "error": error}))
                        break
                    await ws.send_bytes(jpeg_bytes)
            except FastAPIWebSocketDisconnect:
                return
            finally:
                try:
                    recv_task.cancel()
                except Exception:
                    pass

        config = uvicorn.Config(
            app,
            host=self._config.host,
            port=self._config.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._server.run()

    async def _frame_publisher(self, loop: asyncio.AbstractEventLoop, frame_queue: "asyncio.Queue[Optional[Any]]") -> None:
        tick_s = 1.0 / max(1, int(self._config.fps))
        next_tick = time.monotonic()
        while True:
            frame = self._frame_source()

            def _enqueue() -> None:
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                try:
                    frame_queue.put_nowait(frame)
                except asyncio.QueueFull:
                    return

            loop.call_soon_threadsafe(_enqueue)
            next_tick += tick_s
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))


def _apply_ws_input(payload: Mapping[str, Any], key_state: KeyState) -> None:
    event_type = payload.get("type")
    key = payload.get("key")
    if event_type == "keydown" and isinstance(key, str):
        key_state.set_key(key, True)
        return
    if event_type == "keyup" and isinstance(key, str):
        key_state.set_key(key, False)
        return
    keys_payload = payload.get("keys")
    if isinstance(keys_payload, dict):
        key_state.update_from_payload(keys_payload)
        return
    key_state.update_from_payload(payload)


__all__ = ["AttachedServerConfig", "RetroAttachedWebSocketServer"]
