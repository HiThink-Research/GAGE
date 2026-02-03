"""WebRTC server for streaming stable-retro frames and accepting keyboard input."""

from __future__ import annotations

import argparse
import io
import asyncio
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from loguru import logger

from gage_eval.role.arena.games.retro.keyboard_input import KeyState, build_default_key_map
from gage_eval.role.arena.games.retro.retro_env import DEFAULT_PLAYER_ID, StableRetroArenaEnvironment
from gage_eval.role.arena.types import ArenaAction

try:
    from fastapi import Request as FastAPIRequest
    from fastapi import WebSocket as FastAPIWebSocket
    from fastapi import WebSocketDisconnect as FastAPIWebSocketDisconnect
except Exception:  # pragma: no cover - optional import for runtime
    FastAPIRequest = Any  # type: ignore[misc,assignment]
    FastAPIWebSocket = Any  # type: ignore[misc,assignment]
    FastAPIWebSocketDisconnect = Exception  # type: ignore[misc,assignment]

try:
    import numpy as np
except Exception:  # pragma: no cover - optional import for runtime
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional import for runtime
    Image = None
try:
    from fastapi import Request as FastAPIRequest
except Exception:  # pragma: no cover - optional import for runtime
    FastAPIRequest = Any  # type: ignore[misc,assignment]


_CLIENT_DIR = Path(__file__).with_name("webrtc_client")
_INDEX_PATH = _CLIENT_DIR / "index.html"
_CLIENT_JS_PATH = _CLIENT_DIR / "client.js"


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for the retro WebRTC server."""

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
        self._last_frame: Optional[Any] = None
        self._frame_lock = threading.Lock()

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

    def get_latest_frame(self) -> Optional[Any]:
        """Return the latest RGB frame, if available."""

        with self._frame_lock:
            return self._last_frame

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
                metadata={"source": "webrtc"},
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

        with self._frame_lock:
            self._last_frame = frame

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


def _parse_key_payload(message: object) -> Optional[Mapping[str, object]]:
    if isinstance(message, bytes):
        text = message.decode("utf-8", errors="ignore")
    else:
        text = str(message)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _load_web_deps() -> Tuple[Any, Any, Any, Any, Any]:
    try:
        from fastapi import FastAPI
        from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
        from aiortc import RTCPeerConnection, RTCSessionDescription
    except ImportError as exc:
        raise ImportError(
            "WebRTC server dependencies missing. Install aiortc, av, fastapi, uvicorn, and numpy."
        ) from exc
    return FastAPI, FileResponse, HTMLResponse, PlainTextResponse, RTCPeerConnection, RTCSessionDescription


def _encode_jpeg(
    frame: Any,
    *,
    quality: int = 80,
) -> tuple[Optional[bytes], Optional[str]]:
    """Encode an RGB frame into JPEG bytes.

    Args:
        frame: RGB frame as a numpy array.
        quality: JPEG quality setting (1-95).

    Returns:
        Tuple of (jpeg_bytes, error_message).
    """

    if Image is None:
        return None, "PIL_missing"
    if np is None:
        return None, "numpy_missing"
    if not hasattr(frame, "shape"):
        return None, "frame_missing_shape"
    try:
        # STEP 1: Build the base image from the RGB array.
        image = Image.fromarray(frame, mode="RGB")
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"jpeg_encode_failed:{exc}"

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=True)
    return buffer.getvalue(), None


def _apply_ws_input(payload: Mapping[str, Any], key_state: KeyState) -> None:
    """Apply WebSocket input payloads to the key state."""

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


def create_app(config: ServerConfig):
    """Create the FastAPI application for the retro WebRTC server."""

    FastAPI, FileResponse, HTMLResponse, PlainTextResponse, RTCPeerConnection, RTCSessionDescription = _load_web_deps()
    from gage_eval.role.arena.games.retro.webrtc_tracks import GameVideoTrack

    app = FastAPI()
    pcs: set[Any] = set()
    frame_queue: "asyncio.Queue[Optional[Any]]" = asyncio.Queue(maxsize=config.frame_queue_size)
    key_map = build_default_key_map()
    key_state = KeyState(key_map, legal_moves=config.legal_moves)
    game_loop = RetroGameLoop(config=config, frame_queue=frame_queue, key_state=key_state)

    @app.on_event("startup")
    async def _startup() -> None:
        loop = asyncio.get_running_loop()
        game_loop.start(loop)
        logger.info("Retro WebRTC server started for game {}", config.game)
        logger.info("Retro stream UI: http://localhost:{}/", config.port)
        logger.info("Keyboard mapping (key -> action): {}", key_map)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        for pc in pcs:
            await pc.close()
        pcs.clear()
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

    @app.post("/offer")
    async def offer(request: FastAPIRequest) -> Any:
        try:
            logger.info("WebRTC offer request received")
            raw_body = await request.body()
            raw_text = raw_body.decode("utf-8", errors="ignore")
            logger.info("WebRTC offer raw payload (truncated): {}", raw_text[:800])
            params = json.loads(raw_text)
        except Exception as exc:
            logger.warning("WebRTC offer JSON parse failed: {}", exc)
            return PlainTextResponse("Invalid offer payload", status_code=400)

        if not isinstance(params, dict):
            return PlainTextResponse("Invalid offer payload", status_code=400)

        sdp = params.get("sdp")
        sdp_type = params.get("type")
        if not isinstance(sdp, str) or not isinstance(sdp_type, str):
            return PlainTextResponse("Invalid offer payload", status_code=400)

        logger.debug("Received offer with {} chars", len(sdp))
        offer = RTCSessionDescription(sdp=sdp, type=sdp_type)

        pc = RTCPeerConnection()
        pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            logger.info("Peer connection state: {}", pc.connectionState)
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                await pc.close()
                pcs.discard(pc)

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange() -> None:
            logger.info("ICE connection state: {}", pc.iceConnectionState)

        @pc.on("datachannel")
        def on_datachannel(channel) -> None:
            @channel.on("message")
            def on_message(message) -> None:
                payload = _parse_key_payload(message)
                if not payload:
                    return
                if payload.get("reset"):
                    game_loop.request_reset()
                keys_payload = payload.get("keys")
                if isinstance(keys_payload, dict):
                    key_state.update_from_payload(keys_payload)
                else:
                    key_state.update_from_payload(payload)

        pc.addTrack(GameVideoTrack(frame_queue, fps=config.fps))
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        logger.info(
            "WebRTC answer ready type={} sdp_len={}",
            pc.localDescription.type,
            len(pc.localDescription.sdp or ""),
        )

        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    @app.websocket("/ws")
    async def ws_endpoint(ws: FastAPIWebSocket) -> None:
        """Stream JPEG frames and accept keyboard input over WebSocket."""

        await ws.accept()

        async def _recv_loop() -> None:
            while True:
                message = await ws.receive_text()
                payload = _parse_key_payload(message)
                if not payload:
                    continue
                _apply_ws_input(payload, key_state)

        recv_task = asyncio.create_task(_recv_loop())
        try:
            while True:
                # STEP 1: Fetch the latest frame or wait briefly.
                frame = game_loop.get_latest_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                # STEP 2: Encode and stream the JPEG frame.
                jpeg_bytes, error = _encode_jpeg(frame, quality=80)
                if error:
                    await ws.send_text(json.dumps({"ok": False, "error": error}))
                    break
                await ws.send_bytes(jpeg_bytes)
                await asyncio.sleep(1 / max(1, int(config.fps)))
        except FastAPIWebSocketDisconnect:
            return
        finally:
            recv_task.cancel()

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the retro WebRTC server."""

    parser = argparse.ArgumentParser(description="Serve stable-retro frames over WebRTC.")
    parser.add_argument("--game", default="SuperMarioBros3-Nes-v0", help="Retro game id.")
    parser.add_argument("--state", default="Start", help="Retro state name.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=5800, help="Bind port.")
    parser.add_argument("--fps", type=int, default=60, help="Target frames per second.")
    parser.add_argument("--frame-queue-size", type=int, default=2, help="Frame queue size.")
    return parser


def main() -> None:
    """Run the retro WebRTC server."""

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
        raise ImportError("uvicorn is required to run the WebRTC server.") from exc

    logger.info("Uvicorn running on http://localhost:{}/", config.port)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
