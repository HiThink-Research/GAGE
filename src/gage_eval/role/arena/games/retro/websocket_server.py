"""WebSocket server for streaming stable-retro frames and accepting keyboard input."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from loguru import logger

from gage_eval.role.arena.games.retro.keyboard_input import KeyState, build_default_key_map
from gage_eval.role.arena.games.retro.retro_env import DEFAULT_PLAYER_ID, StableRetroArenaEnvironment
from gage_eval.role.arena.games.retro.websocket_tracks import encode_jpeg, parse_key_payload
from gage_eval.role.arena.types import ArenaAction

try:
    from fastapi import WebSocket as FastAPIWebSocket
    from fastapi import WebSocketDisconnect as FastAPIWebSocketDisconnect
except Exception:  # pragma: no cover - optional import for runtime
    FastAPIWebSocket = Any  # type: ignore[misc,assignment]
    FastAPIWebSocketDisconnect = Exception  # type: ignore[misc,assignment]


_CLIENT_DIR = Path(__file__).with_name("websocket_client")
_INDEX_PATH = _CLIENT_DIR / "index.html"
_CLIENT_JS_PATH = _CLIENT_DIR / "client.js"


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for the retro WebSocket server."""

    game: str = "SuperMarioBros3-Nes-v0"
    state: str = "Start"
    host: str = "0.0.0.0"
    port: int = 5800
    fps: int = 60
    frame_queue_size: int = 2
    display_mode: str = "headless"
    legal_moves: Optional[Sequence[str]] = None


class RetroGameLoop:
    """Background loop that advances the retro environment and publishes frames."""

    def __init__(
        self,
        *,
        config: ServerConfig,
        frame_queue: "asyncio.Queue[Optional[Any]]",
        key_state: KeyState,
    ) -> None:
        self._config = config
        self._frame_queue = frame_queue
        self._key_state = key_state
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._reset_event = threading.Event()
        self._env: Optional[StableRetroArenaEnvironment] = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the game loop on a background thread."""

        if self._thread and self._thread.is_alive():
            return
        self._loop = loop
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the game loop and wait for the thread to finish."""

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def request_reset(self) -> None:
        """Request a reset on the next tick."""

        self._reset_event.set()

    def _run(self) -> None:
        # STEP 1: Initialize the retro environment.
        self._env = StableRetroArenaEnvironment(
            game=self._config.game,
            state=self._config.state,
            display_mode=self._config.display_mode,
            legal_moves=self._config.legal_moves,
        )
        self._env.reset()
        self._publish_frame(self._env.get_last_frame())

        # STEP 2: Advance the environment on a fixed timestep.
        tick_s = 1.0 / max(1, int(self._config.fps))
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            if self._reset_event.is_set() and self._env is not None:
                self._env.reset()
                self._reset_event.clear()

            move = self._key_state.resolve_move()
            action = ArenaAction(
                player=DEFAULT_PLAYER_ID,
                move=move,
                raw=move,
                metadata={"source": "websocket"},
            )
            if self._env is not None:
                self._env.apply(action)
                self._publish_frame(self._env.get_last_frame())
                if self._env.is_terminal():
                    self._env.reset()

            next_tick += tick_s
            sleep_for = max(0.0, next_tick - time.monotonic())
            time.sleep(sleep_for)

    def _publish_frame(self, frame: Optional[Any]) -> None:
        if self._loop is None:
            return

        def _enqueue() -> None:
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
            try:
                self._frame_queue.put_nowait(frame)
            except asyncio.QueueFull:
                return

        self._loop.call_soon_threadsafe(_enqueue)


def _apply_ws_input(payload: Mapping[str, Any], key_state: KeyState, game_loop: RetroGameLoop) -> None:
    event_type = payload.get("type")
    if payload.get("reset"):
        game_loop.request_reset()
        return

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


def _load_web_deps() -> tuple[Any, Any, Any, Any]:
    try:
        from fastapi import FastAPI
        from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
    except ImportError as exc:
        raise ImportError(
            "WebSocket server dependencies missing. Install fastapi, uvicorn, numpy, and pillow."
        ) from exc
    return FastAPI, FileResponse, HTMLResponse, PlainTextResponse


def create_app(config: ServerConfig) -> Any:
    """Create the FastAPI application for the retro WebSocket server."""

    FastAPI, FileResponse, HTMLResponse, PlainTextResponse = _load_web_deps()

    app = FastAPI()
    frame_queue: "asyncio.Queue[Optional[Any]]" = asyncio.Queue(maxsize=config.frame_queue_size)
    key_map = build_default_key_map()
    key_state = KeyState(key_map, legal_moves=config.legal_moves)
    game_loop = RetroGameLoop(config=config, frame_queue=frame_queue, key_state=key_state)

    @app.on_event("startup")
    async def _startup() -> None:
        loop = asyncio.get_running_loop()
        game_loop.start(loop)
        logger.info("Retro WebSocket server started for game {}", config.game)
        logger.info("Retro stream UI: http://localhost:{}/", config.port)
        logger.info("Keyboard mapping (key -> action): {}", key_map)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        game_loop.stop()

    @app.get("/")
    async def index() -> Any:
        if not _INDEX_PATH.exists():
            return PlainTextResponse("Missing client bundle", status_code=404)
        return HTMLResponse(_INDEX_PATH.read_text(encoding="utf-8"))

    @app.get("/client.js")
    async def client_js() -> Any:
        if not _CLIENT_JS_PATH.exists():
            return PlainTextResponse("Missing client bundle", status_code=404)
        return FileResponse(_CLIENT_JS_PATH)

    @app.websocket("/ws")
    async def ws_endpoint(ws: FastAPIWebSocket) -> None:
        """Stream JPEG frames and accept keyboard input over WebSocket."""

        await ws.accept()

        async def _recv_loop() -> None:
            while True:
                message = await ws.receive_text()
                payload = parse_key_payload(message)
                if not payload:
                    continue
                _apply_ws_input(payload, key_state, game_loop)

        recv_task = asyncio.create_task(_recv_loop())
        try:
            while True:
                # STEP 1: Await the next frame from the producer.
                frame = await frame_queue.get()
                if frame is None:
                    continue

                # STEP 2: Encode and stream the JPEG frame.
                jpeg_bytes, error = encode_jpeg(frame, quality=80)
                if error:
                    await ws.send_text(json.dumps({"ok": False, "error": error}))
                    break
                await ws.send_bytes(jpeg_bytes)
        except FastAPIWebSocketDisconnect:
            return
        finally:
            recv_task.cancel()

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the retro WebSocket server."""

    parser = argparse.ArgumentParser(description="Serve stable-retro frames over WebSocket.")
    parser.add_argument("--game", default="SuperMarioBros3-Nes-v0", help="Retro game id.")
    parser.add_argument("--state", default="Start", help="Retro state name.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=5800, help="Bind port.")
    parser.add_argument("--fps", type=int, default=60, help="Target frames per second.")
    parser.add_argument("--frame-queue-size", type=int, default=2, help="Frame queue size.")
    return parser


def main() -> None:
    """Run the retro WebSocket server."""

    parser = build_arg_parser()
    args = parser.parse_args()
    config = ServerConfig(
        game=str(args.game),
        state=str(args.state),
        host=str(args.host),
        port=int(args.port),
        fps=int(args.fps),
        frame_queue_size=int(args.frame_queue_size),
    )
    app = create_app(config)

    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("uvicorn is required to run the WebSocket server.") from exc

    logger.info("Uvicorn running on http://localhost:{}/", config.port)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
