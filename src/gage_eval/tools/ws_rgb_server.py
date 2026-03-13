"""In-process RGB hub primitives with optional HTTP control endpoints."""

from __future__ import annotations

import io
import json
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import parse_qs, urlparse

from gage_eval.role.arena.input_mapping import GameInputMapper, HumanActionEvent

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None


_FRAME_RGB_KEYS = ("_rgb", "rgb", "rgb_array", "frame_rgb")
_FRAME_IMAGE_PATH_KEYS = ("_image_path_abs", "image_path", "frame_image_path")
_FRAME_INTERNAL_KEYS = {"_rgb", "_image_path_abs"}


def _is_client_disconnect_error(error: BaseException) -> bool:
    """Return whether an error indicates client-side connection close."""

    if isinstance(error, (BrokenPipeError, ConnectionResetError)):
        return True
    if isinstance(error, OSError):
        return int(getattr(error, "errno", -1)) in {32, 104}
    return False


def _sanitize_frame_payload(frame: Any) -> Any:
    """Remove internal image-only fields before returning JSON payload."""

    if not isinstance(frame, Mapping):
        shape = getattr(frame, "shape", None)
        if shape is None:
            return frame
        try:
            normalized_shape = [int(dim) for dim in shape]
        except Exception:
            normalized_shape = []
        return {"frame_type": "rgb_array", "shape": normalized_shape}
    sanitized = dict(frame)
    for key in _FRAME_INTERNAL_KEYS:
        sanitized.pop(key, None)
    return sanitized


def _extract_rgb_frame(frame: Any) -> Optional[Any]:
    """Extract RGB candidate from frame payload for JPEG encoding."""

    if isinstance(frame, Mapping):
        for key in _FRAME_RGB_KEYS:
            candidate = frame.get(key)
            if candidate is not None:
                return candidate
        return None
    return frame


def _extract_image_path(frame: Any) -> Optional[str]:
    """Extract image path candidate from frame payload."""

    if not isinstance(frame, Mapping):
        return None
    for key in _FRAME_IMAGE_PATH_KEYS:
        value = frame.get(key)
        if value:
            return str(value)
    image = frame.get("image")
    if isinstance(image, Mapping):
        path = image.get("path")
        if path:
            return str(path)
    return None


def _read_image_file(image_path: str) -> tuple[Optional[bytes], Optional[str], Optional[str]]:
    """Read image bytes from file system path."""

    try:
        path = Path(str(image_path)).expanduser().resolve()
    except Exception:
        return None, None, "frame_image_path_invalid"
    if not path.exists():
        return None, None, "frame_image_not_found"
    try:
        image_bytes = path.read_bytes()
    except Exception:
        return None, None, "frame_image_read_failed"
    suffix = str(path.suffix).lower()
    if suffix == ".png":
        content_type = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        content_type = "image/jpeg"
    else:
        content_type = "application/octet-stream"
    return image_bytes, content_type, None


def _encode_jpeg(frame: Any, *, quality: int = 80) -> tuple[Optional[bytes], Optional[str]]:
    """Encode one RGB frame into JPEG bytes.

    Args:
        frame: Frame payload or RGB array object.
        quality: JPEG quality in range [1, 95].

    Returns:
        A tuple of ``(jpeg_bytes, error)``.
    """

    if Image is None:
        return None, "pillow_missing"
    if frame is None:
        return None, "frame_image_missing"

    # STEP 1: Normalize frame input into an array-like object.
    candidate = frame
    if np is not None and isinstance(candidate, (list, tuple)):
        try:
            candidate = np.asarray(candidate)
        except Exception:
            return None, "frame_image_invalid"

    if not hasattr(candidate, "shape"):
        return None, "frame_image_invalid"
    shape = getattr(candidate, "shape", None)
    try:
        dims = tuple(int(dim) for dim in shape)
    except Exception:
        return None, "frame_image_invalid_shape"
    if not dims:
        return None, "frame_image_invalid_shape"

    # STEP 2: Convert the frame into an RGB PIL image.
    try:
        if len(dims) == 2:
            image = Image.fromarray(candidate).convert("RGB")
        elif len(dims) == 3 and dims[2] in {1, 3, 4}:
            image = Image.fromarray(candidate).convert("RGB")
        else:
            return None, "frame_image_invalid_shape"
    except Exception as exc:  # pragma: no cover - defensive path
        return None, f"jpeg_encode_failed:{exc}"

    # STEP 3: Encode PIL image as JPEG bytes.
    clamped_quality = max(1, min(95, int(quality)))
    buffer = io.BytesIO()
    try:
        image.save(buffer, format="JPEG", quality=clamped_quality, optimize=True)
    except Exception as exc:  # pragma: no cover - defensive path
        return None, f"jpeg_encode_failed:{exc}"
    return buffer.getvalue(), None


@dataclass
class DisplayRegistration:
    """Display registration metadata for WsRgbHubServer."""

    display_id: str
    label: str
    human_player_id: str
    frame_source: Callable[[], Any]
    frame_at: Optional[Callable[[int], Any]] = None
    frame_count: Optional[Callable[[], int]] = None
    input_mapper: Optional[GameInputMapper] = None
    legal_moves: Optional[list[str]] = None
    action_queue: Any = None
    default_context: dict[str, Any] = field(default_factory=dict)


class WsRgbHubServer:
    """Manage display registration, frame pull, and mapped input routing."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 5800, allow_origin: str = "*") -> None:
        """Initialize hub server state.

        Args:
            host: Bind host for HTTP control plane.
            port: Bind port for HTTP control plane. Use ``0`` for auto-assign.
            allow_origin: CORS allow-origin value for HTTP endpoints.
        """

        self._host = str(host)
        self._port = int(port)
        self._allow_origin = str(allow_origin)
        self._bound_host = self._host
        self._bound_port = self._port
        self._running = False
        self._displays: dict[str, DisplayRegistration] = {}
        self._http_server: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the hub and HTTP control plane."""

        if self._running:
            return

        # STEP 1: Bind HTTP server on configured host/port (fallback to ephemeral on conflicts).
        server: Optional[ThreadingHTTPServer] = None
        bind_error: Optional[Exception] = None
        try:
            server = ThreadingHTTPServer((self._host, self._port), _WsRgbRequestHandler)
        except OSError as exc:
            bind_error = exc
            if self._port != 0:
                server = ThreadingHTTPServer((self._host, 0), _WsRgbRequestHandler)
        if server is None:
            raise RuntimeError(f"failed_to_bind_ws_rgb_http_server:{bind_error}") from bind_error

        setattr(server, "ws_rgb_hub", self)
        setattr(server, "allow_origin", self._allow_origin)

        # STEP 2: Start background serving thread.
        thread = threading.Thread(
            target=server.serve_forever,
            name="ws-rgb-hub-http",
            daemon=True,
        )
        thread.start()
        bound_host, bound_port = server.server_address[:2]
        self._bound_host = str(bound_host)
        self._bound_port = int(bound_port)
        self._http_server = server
        self._http_thread = thread
        self._running = True

    def stop(self) -> None:
        """Stop the hub and clear runtime-only display bindings."""

        # STEP 1: Shutdown embedded HTTP server if present.
        if self._http_server is not None:
            try:
                self._http_server.shutdown()
                self._http_server.server_close()
            finally:
                self._http_server = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=1.0)
            self._http_thread = None

        # STEP 2: Reset runtime state.
        self._running = False
        self._displays.clear()

    def register_display(self, registration: DisplayRegistration) -> None:
        """Register or replace one display descriptor."""

        display_id = str(registration.display_id)
        if not display_id:
            raise ValueError("display_id is required")
        self._displays[display_id] = registration

    def unregister_display(self, display_id: str) -> None:
        """Unregister one display by id."""

        self._displays.pop(str(display_id), None)

    def list_displays(self) -> list[dict[str, Any]]:
        """List all display descriptors."""

        items: list[dict[str, Any]] = []
        for display_id in sorted(self._displays.keys()):
            reg = self._displays[display_id]
            replay_seekable = callable(reg.frame_at) and callable(reg.frame_count)
            replay_total: Optional[int] = None
            if replay_seekable:
                try:
                    replay_total = max(0, int(reg.frame_count()))
                except Exception:
                    replay_total = None
            items.append(
                {
                    "display_id": reg.display_id,
                    "label": reg.label,
                    "human_player_id": reg.human_player_id,
                    "legal_moves": list(reg.legal_moves or []),
                    "accepts_input": reg.input_mapper is not None,
                    "replay_seekable": replay_seekable,
                    "replay_total": replay_total,
                    "running": self._running,
                }
            )
        return items

    @property
    def host(self) -> str:
        """Return the bound host for HTTP endpoints."""

        return self._bound_host

    @property
    def port(self) -> int:
        """Return the bound port for HTTP endpoints."""

        return int(self._bound_port)

    @property
    def base_url(self) -> str:
        """Return HTTP base URL for external clients."""

        return f"http://{self.host}:{self.port}"

    @property
    def viewer_url(self) -> str:
        """Return browser viewer URL for live ws_rgb inspection."""

        return f"{self.base_url}/ws_rgb/viewer"

    def broadcast_frame(self, display_id: str, *, replay_index: Optional[int] = None) -> dict[str, Any]:
        """Fetch one latest frame payload from display frame source."""

        registration = self._displays.get(str(display_id))
        if registration is None:
            return {"ok": False, "error": "display_not_found", "display_id": str(display_id)}
        frame, error = self._resolve_frame_payload(registration, replay_index=replay_index)
        if error:
            return {"ok": False, "error": error, "display_id": registration.display_id}
        sanitized = _sanitize_frame_payload(frame)
        return {"ok": True, "display_id": registration.display_id, "frame": sanitized}

    def broadcast_frame_image(
        self,
        display_id: str,
        *,
        quality: int = 80,
        replay_index: Optional[int] = None,
    ) -> dict[str, Any]:
        """Fetch and encode one latest frame image as JPEG bytes."""

        registration = self._displays.get(str(display_id))
        if registration is None:
            return {"ok": False, "error": "display_not_found", "display_id": str(display_id)}
        frame_payload, error = self._resolve_frame_payload(registration, replay_index=replay_index)
        if error:
            return {"ok": False, "error": error, "display_id": registration.display_id}
        frame_image_path = _extract_image_path(frame_payload)
        if frame_image_path:
            image_bytes, content_type, error = _read_image_file(frame_image_path)
            if error:
                return {"ok": False, "error": error, "display_id": registration.display_id}
            return {
                "ok": True,
                "display_id": registration.display_id,
                "content_type": content_type or "application/octet-stream",
                "image_bytes": image_bytes,
            }
        frame_image = _extract_rgb_frame(frame_payload)
        jpeg_bytes, error = _encode_jpeg(frame_image, quality=quality)
        if error:
            return {"ok": False, "error": error, "display_id": registration.display_id}
        return {
            "ok": True,
            "display_id": registration.display_id,
            "content_type": "image/jpeg",
            "image_bytes": jpeg_bytes,
        }

    def load_replay_frames(self, display_id: str, *, limit: int = 0) -> dict[str, Any]:
        """Load replay frames for one display when indexed replay is supported."""

        registration = self._displays.get(str(display_id))
        if registration is None:
            return {"ok": False, "error": "display_not_found", "display_id": str(display_id)}
        if not callable(registration.frame_at) or not callable(registration.frame_count):
            return {"ok": False, "error": "replay_buffer_unsupported", "display_id": registration.display_id}
        try:
            total = max(0, int(registration.frame_count()))
        except Exception:
            return {"ok": False, "error": "replay_count_failed", "display_id": registration.display_id}

        # STEP 1: Resolve requested load window against available replay size.
        requested = max(0, int(limit))
        load_total = total if requested <= 0 else min(total, requested)

        # STEP 2: Materialize and sanitize replay frames for browser consumption.
        frames: list[Any] = []
        for index in range(load_total):
            try:
                raw_frame = registration.frame_at(index)
            except Exception:
                return {
                    "ok": False,
                    "error": "replay_frame_fetch_failed",
                    "display_id": registration.display_id,
                    "index": index,
                }
            frames.append(_sanitize_frame_payload(raw_frame))
        return {
            "ok": True,
            "display_id": registration.display_id,
            "total": total,
            "loaded": load_total,
            "frames": frames,
        }

    @staticmethod
    def _resolve_frame_payload(
        registration: DisplayRegistration,
        *,
        replay_index: Optional[int],
    ) -> tuple[Any, Optional[str]]:
        if replay_index is None:
            return registration.frame_source(), None
        frame_at = registration.frame_at
        if not callable(frame_at):
            return None, "replay_index_unsupported"
        try:
            normalized_index = max(0, int(replay_index))
        except (TypeError, ValueError):
            return None, "invalid_replay_index"
        try:
            return frame_at(normalized_index), None
        except Exception:
            return None, "replay_index_fetch_failed"

    def handle_input(
        self,
        *,
        display_id: str,
        payload: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Route browser input through mapper and enqueue mapped actions."""

        registration = self._displays.get(str(display_id))
        if registration is None:
            return {"ok": False, "error": "display_not_found", "display_id": str(display_id)}
        mapper = registration.input_mapper
        if mapper is None:
            return {"ok": False, "error": "input_mapper_missing", "display_id": registration.display_id}

        merged_context = dict(registration.default_context)
        merged_context.setdefault("display_id", registration.display_id)
        merged_context.setdefault("human_player_id", registration.human_player_id)
        if context:
            merged_context.update(dict(context))
        actions = mapper.handle_browser_event(payload, context=merged_context)
        queued = self._enqueue_actions(registration.action_queue, actions)
        return {
            "ok": True,
            "display_id": registration.display_id,
            "queued": queued,
            "actions": [item.to_dict() for item in actions],
        }

    @staticmethod
    def _enqueue_actions(action_queue: Any, actions: list[HumanActionEvent]) -> int:
        if action_queue is None:
            return 0
        queued = 0
        for action in actions:
            payload = json.dumps(
                {
                    "player_id": str(action.player_id or ""),
                    "move": str(action.move or ""),
                    "raw": str(action.raw or action.move or ""),
                    "metadata": dict(action.metadata or {}),
                },
                ensure_ascii=False,
            )
            if hasattr(action_queue, "put_nowait"):
                action_queue.put_nowait(payload)
            else:
                action_queue.put(payload)
            queued += 1
        return queued


class _WsRgbRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for WsRgbHubServer control APIs."""

    server_version = "GAGEWsRgbHub/1.0"
    _VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GAGE Game Play Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1220;
      --panel: #141e30;
      --panel-alt: #1b263d;
      --text: #d6deed;
      --muted: #91a0bc;
      --accent: #5bc0be;
      --warn: #ffcc66;
      --error: #ff7d7d;
      --ok: #85d988;
      --border: #2b3a58;
      --viewer-left-width: 300px;
      --viewer-right-width: 340px;
      --viewer-splitter-width: 14px;
    }
    * {
      box-sizing: border-box;
      font-family: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }
    body {
      margin: 0;
      background: linear-gradient(165deg, var(--bg) 0%, #0e1a2a 45%, #09111b 100%);
      color: var(--text);
      min-height: 100vh;
      padding: 12px;
    }
    body.split-resizing {
      cursor: col-resize;
      user-select: none;
    }
    .shell {
      width: min(1400px, calc(100vw - 24px));
      max-width: 1400px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    .panel {
      background: linear-gradient(180deg, var(--panel) 0%, var(--panel-alt) 100%);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      box-shadow: 0 6px 22px rgba(0, 0, 0, 0.35);
      min-width: 0;
    }
    .title {
      font-size: 15px;
      font-weight: 700;
      margin-bottom: 8px;
      color: var(--accent);
    }
    .row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .row + .row {
      margin-top: 8px;
    }
    select, input, button {
      background: #0f1727;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 8px;
    }
    select {
      min-width: 280px;
    }
    select.compact {
      min-width: 120px;
    }
    input {
      min-width: 120px;
    }
    input[type="range"] {
      min-width: 220px;
    }
    button {
      cursor: pointer;
    }
    button:hover {
      border-color: var(--accent);
    }
    .status {
      color: var(--muted);
      font-size: 12px;
      white-space: pre-wrap;
    }
    .status.ok {
      color: var(--ok);
    }
    .status.warn {
      color: var(--warn);
    }
    .status.error {
      color: var(--error);
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #0b1322;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      max-height: 360px;
      overflow: auto;
      line-height: 1.45;
      font-size: 12px;
    }
    .frame-image-shell {
      width: 100%;
      aspect-ratio: 4 / 3;
      min-height: clamp(260px, 28vw, 520px);
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #0b1322;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .frame-image-shell img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: none;
    }
    .frame-image-hint {
      margin-top: 8px;
    }
    .split {
      display: grid;
      grid-template-columns:
        minmax(220px, var(--viewer-left-width))
        var(--viewer-splitter-width)
        minmax(480px, 1fr)
        var(--viewer-splitter-width)
        minmax(240px, var(--viewer-right-width));
      gap: 12px;
      align-items: stretch;
    }
    .splitter {
      position: relative;
      width: 100%;
      border-radius: 999px;
      cursor: col-resize;
      touch-action: none;
      background: linear-gradient(180deg, rgba(91, 192, 190, 0.16) 0%, rgba(43, 58, 88, 0.75) 100%);
      box-shadow: inset 0 0 0 1px rgba(91, 192, 190, 0.18);
      transition: background 120ms ease, box-shadow 120ms ease;
    }
    .splitter::before {
      content: "";
      position: absolute;
      inset: 14px 4px;
      border-radius: 999px;
      background: linear-gradient(180deg, rgba(91, 192, 190, 0.85) 0%, rgba(214, 222, 237, 0.3) 100%);
      opacity: 0.92;
    }
    .splitter:hover,
    .splitter:focus-visible {
      background: linear-gradient(180deg, rgba(91, 192, 190, 0.28) 0%, rgba(43, 58, 88, 0.9) 100%);
      box-shadow: inset 0 0 0 1px rgba(91, 192, 190, 0.4), 0 0 0 1px rgba(91, 192, 190, 0.18);
      outline: none;
    }
    .panel-image {
      display: flex;
      flex-direction: column;
    }
    .panel-image .frame-image-shell {
      flex: 1 1 auto;
    }
    @media (max-width: 1100px) {
      .split {
        grid-template-columns: 1fr;
      }
      .splitter {
        display: none;
      }
      .frame-image-shell {
        min-height: clamp(280px, 56vw, 520px);
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="panel">
      <div class="title">GAGE Game Play Viewer</div>
      <div class="row">
        <label for="displaySelect">Display:</label>
        <select id="displaySelect"></select>
        <button id="refreshDisplaysBtn" type="button">Refresh Displays</button>
      </div>
      <div class="row">
        <label for="pollMsInput">Poll ms:</label>
        <input id="pollMsInput" type="number" min="20" max="2000" step="10" value="20" />
        <button id="applyPollBtn" type="button">Apply</button>
        <button id="fetchFrameBtn" type="button">Fetch Once</button>
      </div>
      <div class="row">
        <label for="actionInput">Action:</label>
        <input id="actionInput" type="text" value="0" />
        <button id="sendActionBtn" type="button">Send Action</button>
        <label><input id="keyCaptureToggle" type="checkbox" /> Key Capture</label>
      </div>
      <div class="row">
        <button id="captureToggleBtn" type="button">Capture: ON</button>
        <button id="replayToggleBtn" type="button">Start Replay</button>
        <button id="replayStepPrevBtn" type="button">Step -</button>
        <button id="replayStepNextBtn" type="button">Step +</button>
        <button id="replayLiveBtn" type="button">Back To Live</button>
      </div>
      <div class="row">
        <label for="replaySpeedSelect">Replay Speed:</label>
        <select id="replaySpeedSelect" class="compact">
          <option value="0.5">0.5x</option>
          <option value="1" selected>1.0x</option>
          <option value="2">2.0x</option>
          <option value="4">4.0x</option>
        </select>
        <label for="replaySeek">Replay Seek:</label>
        <input id="replaySeek" type="range" min="0" max="0" step="1" value="0" />
      </div>
      <div id="replayStatus" class="status">Replay buffer: 0 frame(s)</div>
      <div id="status" class="status">Initializing...</div>
    </div>

    <div class="split" id="viewerSplit">
      <div class="panel">
        <div class="title">Frame Text</div>
        <pre id="boardText">(waiting frame)</pre>
      </div>
      <div
        class="splitter"
        id="splitterLeft"
        role="separator"
        aria-label="Resize frame text and frame image panels"
        tabindex="0"
      ></div>
      <div class="panel panel-image">
        <div class="title">Frame Image (RGB)</div>
        <div class="frame-image-shell">
          <img id="frameImage" alt="ws_rgb_frame_image" />
        </div>
        <div id="frameImageHint" class="status frame-image-hint">No RGB frame available yet.</div>
      </div>
      <div
        class="splitter"
        id="splitterRight"
        role="separator"
        aria-label="Resize frame image and frame JSON panels"
        tabindex="0"
      ></div>
      <div class="panel">
        <div class="title">Frame JSON</div>
        <pre id="frameJson">{}</pre>
      </div>
    </div>
  </div>

  <script>
    const state = {
      displays: [],
      selectedDisplayId: "",
      selectedDisplayAcceptsInput: false,
      selectedDisplayReplaySeekable: false,
      pollTimer: null,
      pollMs: 20,
      liveFrameImageUrl: "",
      captureEnabled: true,
      historyFrames: [],
      historyLimit: 2000,
      replayMode: false,
      replayIndex: -1,
      replayTimer: null,
      replaySpeed: 1.0,
      lastSnapshotDisplayId: "",
      lastSnapshotSignature: "",
      captureCount: 0,
      replaySeekDragging: false,
      replayBufferLoading: false,
      replayBufferLoadedForDisplay: "",
      panelLayout: { leftWidth: 300, rightWidth: 340 },
      panelResizeDrag: null,
    };

    const PANEL_LAYOUT_STORAGE_KEY = "gage_ws_rgb_panel_layout_v2";
    const PANEL_LAYOUT_DEFAULTS = {
      leftWidth: 300,
      rightWidth: 340,
      minLeftWidth: 220,
      minRightWidth: 240,
      minCenterWidth: 480,
      keyboardStep: 24,
    };

    const el = {
      viewerSplit: document.getElementById("viewerSplit"),
      displaySelect: document.getElementById("displaySelect"),
      refreshDisplaysBtn: document.getElementById("refreshDisplaysBtn"),
      pollMsInput: document.getElementById("pollMsInput"),
      applyPollBtn: document.getElementById("applyPollBtn"),
      fetchFrameBtn: document.getElementById("fetchFrameBtn"),
      actionInput: document.getElementById("actionInput"),
      sendActionBtn: document.getElementById("sendActionBtn"),
      keyCaptureToggle: document.getElementById("keyCaptureToggle"),
      captureToggleBtn: document.getElementById("captureToggleBtn"),
      replayToggleBtn: document.getElementById("replayToggleBtn"),
      replayStepPrevBtn: document.getElementById("replayStepPrevBtn"),
      replayStepNextBtn: document.getElementById("replayStepNextBtn"),
      replayLiveBtn: document.getElementById("replayLiveBtn"),
      replaySpeedSelect: document.getElementById("replaySpeedSelect"),
      replaySeek: document.getElementById("replaySeek"),
      replayStatus: document.getElementById("replayStatus"),
      status: document.getElementById("status"),
      boardText: document.getElementById("boardText"),
      frameJson: document.getElementById("frameJson"),
      frameImage: document.getElementById("frameImage"),
      frameImageHint: document.getElementById("frameImageHint"),
      splitterLeft: document.getElementById("splitterLeft"),
      splitterRight: document.getElementById("splitterRight"),
    };

    function setStatus(message, tone = "status") {
      el.status.className = tone;
      el.status.textContent = message;
    }

    function setReplayStatus(message, tone = "status") {
      el.replayStatus.className = tone;
      el.replayStatus.textContent = message;
    }

    function safeCloneFrame(frame) {
      try {
        return JSON.parse(JSON.stringify(frame || {}));
      } catch (_) {
        return frame || {};
      }
    }

    function clamp(value, minValue, maxValue) {
      return Math.min(maxValue, Math.max(minValue, value));
    }

    function readStoredPanelLayout() {
      try {
        const raw = window.localStorage.getItem(PANEL_LAYOUT_STORAGE_KEY);
        if (!raw) {
          return null;
        }
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object") {
          return null;
        }
        return {
          leftWidth: Number(parsed.leftWidth),
          rightWidth: Number(parsed.rightWidth),
        };
      } catch (_) {
        return null;
      }
    }

    function savePanelLayout() {
      try {
        window.localStorage.setItem(PANEL_LAYOUT_STORAGE_KEY, JSON.stringify(state.panelLayout));
      } catch (_) {
        return;
      }
    }

    function isCompactViewerLayout() {
      return window.matchMedia("(max-width: 1100px)").matches;
    }

    function normalizePanelLayout(layout) {
      const next = layout && typeof layout === "object" ? layout : {};
      const containerWidth = el.viewerSplit ? el.viewerSplit.clientWidth : 0;
      const leftBase = Number.isFinite(Number(next.leftWidth))
        ? Number(next.leftWidth)
        : Number(state.panelLayout.leftWidth || PANEL_LAYOUT_DEFAULTS.leftWidth);
      const rightBase = Number.isFinite(Number(next.rightWidth))
        ? Number(next.rightWidth)
        : Number(state.panelLayout.rightWidth || PANEL_LAYOUT_DEFAULTS.rightWidth);
      if (!containerWidth || isCompactViewerLayout()) {
        return {
          leftWidth: clamp(leftBase, PANEL_LAYOUT_DEFAULTS.minLeftWidth, 480),
          rightWidth: clamp(rightBase, PANEL_LAYOUT_DEFAULTS.minRightWidth, 560),
        };
      }
      const splitStyles = getComputedStyle(el.viewerSplit);
      const splitterWidth = Number(getComputedStyle(document.documentElement).getPropertyValue("--viewer-splitter-width").replace("px", "")) || 14;
      const columnGap = Number(splitStyles.columnGap.replace("px", "")) || 0;
      const reservedWidth = PANEL_LAYOUT_DEFAULTS.minCenterWidth + splitterWidth * 2 + columnGap * 4;
      const sideBudget = Math.max(
        PANEL_LAYOUT_DEFAULTS.minLeftWidth + PANEL_LAYOUT_DEFAULTS.minRightWidth,
        containerWidth - reservedWidth,
      );
      let leftWidth = clamp(
        leftBase,
        PANEL_LAYOUT_DEFAULTS.minLeftWidth,
        Math.max(PANEL_LAYOUT_DEFAULTS.minLeftWidth, sideBudget - PANEL_LAYOUT_DEFAULTS.minRightWidth),
      );
      let rightWidth = clamp(
        rightBase,
        PANEL_LAYOUT_DEFAULTS.minRightWidth,
        Math.max(PANEL_LAYOUT_DEFAULTS.minRightWidth, sideBudget - leftWidth),
      );
      leftWidth = clamp(
        leftWidth,
        PANEL_LAYOUT_DEFAULTS.minLeftWidth,
        Math.max(PANEL_LAYOUT_DEFAULTS.minLeftWidth, sideBudget - rightWidth),
      );
      return {
        leftWidth,
        rightWidth,
      };
    }

    function applyPanelLayout(layout, { persist = false } = {}) {
      const normalized = normalizePanelLayout(layout);
      state.panelLayout = normalized;
      if (!el.viewerSplit) {
        return normalized;
      }
      el.viewerSplit.style.setProperty("--viewer-left-width", `${normalized.leftWidth}px`);
      el.viewerSplit.style.setProperty("--viewer-right-width", `${normalized.rightWidth}px`);
      if (persist) {
        savePanelLayout();
      }
      return normalized;
    }

    function resetPanelLayout() {
      applyPanelLayout(
        {
          leftWidth: PANEL_LAYOUT_DEFAULTS.leftWidth,
          rightWidth: PANEL_LAYOUT_DEFAULTS.rightWidth,
        },
        { persist: true },
      );
    }

    function startPanelResize(side, event) {
      if (isCompactViewerLayout()) {
        return;
      }
      event.preventDefault();
      const current = applyPanelLayout(state.panelLayout);
      state.panelResizeDrag = {
        side,
        startX: Number(event.clientX),
        leftWidth: current.leftWidth,
        rightWidth: current.rightWidth,
      };
      document.body.classList.add("split-resizing");
    }

    function handlePanelResize(event) {
      if (!state.panelResizeDrag) {
        return;
      }
      const deltaX = Number(event.clientX) - Number(state.panelResizeDrag.startX);
      if (state.panelResizeDrag.side === "left") {
        applyPanelLayout({
          leftWidth: Number(state.panelResizeDrag.leftWidth) + deltaX,
          rightWidth: state.panelResizeDrag.rightWidth,
        });
        return;
      }
      applyPanelLayout({
        leftWidth: state.panelResizeDrag.leftWidth,
        rightWidth: Number(state.panelResizeDrag.rightWidth) - deltaX,
      });
    }

    function stopPanelResize() {
      if (!state.panelResizeDrag) {
        return;
      }
      state.panelResizeDrag = null;
      document.body.classList.remove("split-resizing");
      savePanelLayout();
    }

    function resizePanelByKeyboard(side, direction) {
      if (direction === 0) {
        return;
      }
      const delta = Number(PANEL_LAYOUT_DEFAULTS.keyboardStep) * direction;
      if (side === "left") {
        applyPanelLayout(
          {
            leftWidth: Number(state.panelLayout.leftWidth) + delta,
            rightWidth: state.panelLayout.rightWidth,
          },
          { persist: true },
        );
        return;
      }
      applyPanelLayout(
        {
          leftWidth: state.panelLayout.leftWidth,
          rightWidth: Number(state.panelLayout.rightWidth) - delta,
        },
        { persist: true },
      );
    }

    function bindPanelResizeEvents() {
      if (!el.splitterLeft || !el.splitterRight) {
        return;
      }
      el.splitterLeft.addEventListener("pointerdown", (event) => {
        startPanelResize("left", event);
      });
      el.splitterRight.addEventListener("pointerdown", (event) => {
        startPanelResize("right", event);
      });
      el.splitterLeft.addEventListener("dblclick", () => {
        resetPanelLayout();
      });
      el.splitterRight.addEventListener("dblclick", () => {
        resetPanelLayout();
      });
      el.splitterLeft.addEventListener("keydown", (event) => {
        if (event.key === "ArrowLeft") {
          event.preventDefault();
          resizePanelByKeyboard("left", -1);
        } else if (event.key === "ArrowRight") {
          event.preventDefault();
          resizePanelByKeyboard("left", 1);
        }
      });
      el.splitterRight.addEventListener("keydown", (event) => {
        if (event.key === "ArrowLeft") {
          event.preventDefault();
          resizePanelByKeyboard("right", -1);
        } else if (event.key === "ArrowRight") {
          event.preventDefault();
          resizePanelByKeyboard("right", 1);
        }
      });
      window.addEventListener("pointermove", handlePanelResize);
      window.addEventListener("pointerup", stopPanelResize);
      window.addEventListener("pointercancel", stopPanelResize);
      window.addEventListener("resize", () => {
        applyPanelLayout(state.panelLayout);
      });
    }

    function getSelectedDisplay() {
      if (!state.selectedDisplayId) {
        return null;
      }
      return state.displays.find((item) => String(item.display_id || "") === state.selectedDisplayId) || null;
    }

    function updateInputUi() {
      const selected = getSelectedDisplay();
      const acceptsInput = Boolean(selected && selected.accepts_input);
      const replaySeekable = Boolean(selected && selected.replay_seekable);
      state.selectedDisplayAcceptsInput = acceptsInput;
      state.selectedDisplayReplaySeekable = replaySeekable;
      const disabled = !state.selectedDisplayId || !acceptsInput;
      el.actionInput.disabled = disabled;
      el.sendActionBtn.disabled = disabled;
      el.keyCaptureToggle.disabled = disabled;
      if (disabled) {
        el.keyCaptureToggle.checked = false;
      }
    }

    function buildReplayImageEndpoint(index) {
      if (!state.selectedDisplayId) {
        return "";
      }
      const query = new URLSearchParams({
        display_id: state.selectedDisplayId,
        replay_index: String(Math.max(0, Number(index || "0"))),
        t: String(Date.now()),
      });
      return `/ws_rgb/frame_image?${query.toString()}`;
    }

    function buildSnapshotSignature(frame) {
      if (!frame || typeof frame !== "object") {
        return String(frame);
      }
      const metadata = frame.metadata && typeof frame.metadata === "object" ? frame.metadata : {};
      const replaySeq = metadata.replay_seq;
      const replayIndex = metadata.replay_index;
      const replayStep = metadata.replay_step;
      if (replaySeq != null || replayIndex != null || replayStep != null) {
        return `replay:${String(replaySeq ?? "")}:${String(replayIndex ?? "")}:${String(replayStep ?? "")}`;
      }
      const moveCount = frame.move_count;
      const lastMove = frame.last_move;
      const boardText = frame.board_text;
      if (moveCount != null || lastMove != null || boardText != null) {
        return `live:${String(moveCount ?? "")}:${String(lastMove ?? "")}:${String(boardText ?? "")}`;
      }
      try {
        return JSON.stringify(frame);
      } catch (_) {
        return String(frame);
      }
    }

    async function apiJson(url, options) {
      const response = await fetch(url, options);
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const err = payload && payload.error ? payload.error : "request_failed";
        throw new Error(err);
      }
      return payload;
    }

    function normalizeAction(raw) {
      const text = String(raw || "").trim();
      if (!text) {
        return "0";
      }
      if (/^-?\\d+$/.test(text)) {
        return Number(text);
      }
      return text;
    }

    function releaseLiveFrameImage() {
      if (!state.liveFrameImageUrl) {
        return;
      }
      URL.revokeObjectURL(state.liveFrameImageUrl);
      state.liveFrameImageUrl = "";
    }

    function clearFrameImage(hint = "No RGB frame available yet.") {
      releaseLiveFrameImage();
      el.frameImage.removeAttribute("src");
      el.frameImage.style.display = "none";
      el.frameImageHint.textContent = hint;
      el.frameImageHint.className = "status frame-image-hint warn";
    }

    function showFrameImageSource(source, hint, tone = "ok") {
      el.frameImage.src = String(source || "");
      el.frameImage.style.display = "block";
      el.frameImageHint.textContent = hint;
      el.frameImageHint.className = `status frame-image-hint ${tone}`;
    }

    function showLiveFrameImage(blob) {
      const imageUrl = URL.createObjectURL(blob);
      releaseLiveFrameImage();
      state.liveFrameImageUrl = imageUrl;
      showFrameImageSource(
        imageUrl,
        `${blob.type || "image/jpeg"} · ${blob.size} bytes`,
        "ok",
      );
    }

    async function blobToDataUrl(blob) {
      return await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(new Error("blob_to_data_url_failed"));
        reader.readAsDataURL(blob);
      });
    }

    function stopReplayPlayback() {
      if (state.replayTimer === null) {
        return;
      }
      clearInterval(state.replayTimer);
      state.replayTimer = null;
    }

    function updateReplayUi() {
      const total = state.historyFrames.length;
      const hasFrames = total > 0;
      const canReplay = total > 1;
      if (!hasFrames) {
        state.replayIndex = -1;
      } else if (state.replayIndex < 0) {
        state.replayIndex = total - 1;
      } else if (state.replayIndex >= total) {
        state.replayIndex = total - 1;
      }

      el.captureToggleBtn.textContent = state.captureEnabled ? "Capture: ON" : "Capture: OFF";
      el.replayToggleBtn.textContent = state.replayTimer === null ? "Start Replay" : "Pause Replay";
      el.replaySeek.max = String(Math.max(0, total - 1));
      if (!state.replaySeekDragging) {
        el.replaySeek.value = String(Math.max(0, state.replayIndex));
      }
      el.replaySeek.disabled = !canReplay;
      el.replayToggleBtn.disabled = !canReplay;
      el.replayStepPrevBtn.disabled = !canReplay;
      el.replayStepNextBtn.disabled = !canReplay;
      el.replayLiveBtn.disabled = !state.replayMode;
      updateInputUi();

      const mode = state.replayMode ? "replay" : "live";
      const globalTotal = Math.max(state.captureCount, total);
      const selectedSnapshot = hasFrames ? state.historyFrames[Math.max(0, state.replayIndex)] : null;
      let currentCaptureIndex = 0;
      if (selectedSnapshot && Number.isFinite(Number(selectedSnapshot.captureIndex))) {
        currentCaptureIndex = Number(selectedSnapshot.captureIndex);
      } else if (hasFrames) {
        currentCaptureIndex = Math.max(1, globalTotal - (total - 1 - state.replayIndex));
      }
      const indexText = hasFrames ? `${currentCaptureIndex}/${globalTotal}` : "0/0";
      const bufferText = hasFrames ? `${state.replayIndex + 1}/${total}` : "0/0";
      const droppedText = state.captureCount > total ? ` | dropped=${state.captureCount - total}` : "";
      const singleFrameText = hasFrames && !canReplay ? " | single_frame_only=true" : "";
      setReplayStatus(
        `Replay buffer: ${total} frame(s) | mode=${mode} | index=${indexText} | buffer=${bufferText}${droppedText}${singleFrameText} | speed=${state.replaySpeed.toFixed(2)}x`,
        state.replayMode ? "status ok" : "status",
      );
    }

    function resetReplayBuffer() {
      stopReplayPlayback();
      state.historyFrames = [];
      state.replayIndex = -1;
      state.replayMode = false;
      state.lastSnapshotDisplayId = "";
      state.lastSnapshotSignature = "";
      state.captureCount = 0;
      state.replayBufferLoadedForDisplay = "";
      updateReplayUi();
    }

    function renderDisplays(displays) {
      state.displays = Array.isArray(displays) ? displays : [];
      const oldSelected = state.selectedDisplayId;
      el.displaySelect.innerHTML = "";
      for (const item of state.displays) {
        const id = String(item.display_id || "");
        if (!id) {
          continue;
        }
        const option = document.createElement("option");
        option.value = id;
        option.textContent = `${id} (${item.label || "display"})`;
        el.displaySelect.appendChild(option);
      }
      if (state.displays.length === 0) {
        state.selectedDisplayId = "";
        state.selectedDisplayAcceptsInput = false;
        state.selectedDisplayReplaySeekable = false;
        resetReplayBuffer();
        updateInputUi();
        clearFrameImage("No display registered yet.");
        return;
      }
      const hasOld = state.displays.some((item) => String(item.display_id || "") === oldSelected);
      state.selectedDisplayId = hasOld ? oldSelected : String(state.displays[0].display_id || "");
      el.displaySelect.value = state.selectedDisplayId;
      updateInputUi();
    }

    async function refreshDisplays() {
      try {
        const payload = await apiJson("/ws_rgb/displays");
        renderDisplays(payload.displays || []);
        if (!state.selectedDisplayId) {
          setStatus("No display registered yet.", "status warn");
          return;
        }
        const mode = state.replayMode ? "replay" : "live";
        const replayFlag = state.selectedDisplayReplaySeekable ? " replay_buffer=available" : "";
        if (state.selectedDisplayAcceptsInput) {
          setStatus(`Connected. display_id=${state.selectedDisplayId} mode=${mode}${replayFlag}`, "status ok");
        } else {
          setStatus(
            `Connected. display_id=${state.selectedDisplayId} mode=${mode}${replayFlag} (input disabled: no input mapper)`,
            "status warn",
          );
        }
      } catch (error) {
        setStatus(`Refresh displays failed: ${error.message}`, "status error");
      }
    }

    async function fetchFrameImageBlob() {
      if (!state.selectedDisplayId) {
        return { ok: false, error: "missing_display_id" };
      }
      const query = new URLSearchParams({
        display_id: state.selectedDisplayId,
        t: String(Date.now()),
      });
      const response = await fetch(`/ws_rgb/frame_image?${query.toString()}`);
      if (response.status === 404) {
        return { ok: false, error: "frame_image_missing" };
      }
      if (!response.ok) {
        let error = `frame_image_request_failed:${response.status}`;
        try {
          const payload = await response.json();
          if (payload && payload.error) {
            error = String(payload.error);
          }
        } catch (_) {
          // Best-effort error extraction from JSON response.
        }
        throw new Error(error);
      }
      const blob = await response.blob();
      if (!blob || blob.size <= 0) {
        return { ok: false, error: "frame_image_empty" };
      }
      return { ok: true, blob };
    }

    async function preloadReplayBuffer() {
      if (!state.selectedDisplayId || !state.selectedDisplayReplaySeekable) {
        return;
      }
      if (state.replayBufferLoading) {
        return;
      }
      state.replayBufferLoading = true;
      try {
        const query = new URLSearchParams({ display_id: state.selectedDisplayId });
        const payload = await apiJson(`/ws_rgb/replay_buffer?${query.toString()}`);
        const frames = Array.isArray(payload.frames) ? payload.frames : [];

        // STEP 1: Replace local replay buffer from server-side replay payload.
        stopReplayPlayback();
        state.historyFrames = frames.map((frame, index) => ({
          frame: safeCloneFrame(frame),
          imageDataUrl: "",
          imageEndpoint: buildReplayImageEndpoint(index),
          imageHint: "Replay image loaded on demand.",
          displayId: state.selectedDisplayId,
          timestampMs: Date.now(),
          captureIndex: index + 1,
        }));
        state.captureCount = state.historyFrames.length;
        state.lastSnapshotDisplayId = "";
        state.lastSnapshotSignature = "";
        state.replayIndex = state.historyFrames.length > 0 ? 0 : -1;
        state.replayMode = state.historyFrames.length > 0;
        state.replayBufferLoadedForDisplay = state.selectedDisplayId;

        // STEP 2: Render first snapshot immediately so user can inspect without waiting.
        if (state.historyFrames.length > 0) {
          applySnapshotToPanels(state.historyFrames[0]);
          updateReplayUi();
          setStatus(`Replay buffer loaded. frames=${state.historyFrames.length}`, "status ok");
        } else {
          updateReplayUi();
          clearFrameImage("No replay frames available.");
          setStatus("Replay buffer is empty.", "status warn");
        }
      } catch (error) {
        setStatus(`Load replay buffer failed: ${error.message}`, "status error");
      } finally {
        state.replayBufferLoading = false;
      }
    }

    function applyFrameToPanels(frame) {
      const boardText = frame.board_text != null ? String(frame.board_text) : "(no text)";
      el.boardText.textContent = boardText;
      el.frameJson.textContent = JSON.stringify(frame, null, 2);
    }

    function applySnapshotToPanels(snapshot) {
      applyFrameToPanels(snapshot.frame || {});
      if (snapshot.imageDataUrl) {
        releaseLiveFrameImage();
        showFrameImageSource(snapshot.imageDataUrl, snapshot.imageHint || "replay image", "ok");
        return;
      }
      if (snapshot.imageEndpoint) {
        releaseLiveFrameImage();
        showFrameImageSource(snapshot.imageEndpoint, snapshot.imageHint || "replay image", "ok");
        return;
      }
      clearFrameImage(snapshot.imageHint || "No RGB frame available in snapshot.");
    }

    function renderReplayIndex(index) {
      if (state.historyFrames.length === 0) {
        updateReplayUi();
        return;
      }
      const clamped = Math.max(0, Math.min(state.historyFrames.length - 1, Number(index)));
      state.replayIndex = clamped;
      applySnapshotToPanels(state.historyFrames[clamped]);
      updateReplayUi();
    }

    function setReplayMode(enabled) {
      const next = Boolean(enabled);
      if (!next) {
        state.replayMode = false;
        stopReplayPlayback();
        updateReplayUi();
        return;
      }
      if (state.historyFrames.length === 0) {
        state.replayMode = false;
        updateReplayUi();
        setStatus("Replay buffer is empty.", "status warn");
        return;
      }
      state.replayMode = true;
      if (state.replayIndex < 0) {
        state.replayIndex = state.historyFrames.length - 1;
      }
      renderReplayIndex(state.replayIndex);
    }

    function startReplayPlayback() {
      if (state.historyFrames.length === 0) {
        setStatus("Replay buffer is empty.", "status warn");
        return;
      }
      if (state.historyFrames.length <= 1) {
        setStatus("Replay requires at least 2 captured frames.", "status warn");
        return;
      }
      setReplayMode(true);
      stopReplayPlayback();
      if (state.replayIndex >= state.historyFrames.length - 1) {
        state.replayIndex = 0;
      }
      const tickMs = Math.max(20, Math.round(state.pollMs / Math.max(0.1, state.replaySpeed)));
      state.replayTimer = setInterval(() => {
        if (state.historyFrames.length === 0) {
          stopReplayPlayback();
          updateReplayUi();
          return;
        }
        const next = state.replayIndex + 1;
        if (next >= state.historyFrames.length) {
          stopReplayPlayback();
          updateReplayUi();
          setStatus("Replay reached end.", "status ok");
          return;
        }
        renderReplayIndex(next);
      }, tickMs);
      updateReplayUi();
    }

    function pushReplaySnapshot(frame, imageDataUrl, imageHint) {
      const snapshotSignature = buildSnapshotSignature(frame);
      if (
        snapshotSignature &&
        state.lastSnapshotDisplayId === state.selectedDisplayId &&
        state.lastSnapshotSignature === snapshotSignature
      ) {
        return;
      }
      state.captureCount += 1;
      const snapshot = {
        frame: safeCloneFrame(frame),
        imageDataUrl: imageDataUrl || "",
        imageHint: imageHint || "No RGB frame available in snapshot.",
        displayId: state.selectedDisplayId,
        timestampMs: Date.now(),
        captureIndex: state.captureCount,
      };
      state.historyFrames.push(snapshot);
      state.lastSnapshotDisplayId = state.selectedDisplayId;
      state.lastSnapshotSignature = snapshotSignature;
      if (state.historyFrames.length > state.historyLimit) {
        state.historyFrames.shift();
        if (state.replayIndex > 0) {
          state.replayIndex -= 1;
        }
      }
      if (!state.replayMode) {
        state.replayIndex = state.historyFrames.length - 1;
      }
      updateReplayUi();
    }

    async function fetchFrame() {
      if (!state.selectedDisplayId) {
        clearFrameImage("No display selected.");
        return;
      }
      if (state.replayMode) {
        return;
      }
      try {
        const query = new URLSearchParams({ display_id: state.selectedDisplayId });
        const payload = await apiJson(`/ws_rgb/frame?${query.toString()}`);
        const frame = payload.frame || {};
        applyFrameToPanels(frame);

        let imagePayload = { ok: false, error: "frame_image_missing" };
        try {
          imagePayload = await fetchFrameImageBlob();
        } catch (error) {
          setStatus(`Fetch frame image failed: ${error.message}`, "status error");
        }

        let snapshotImageDataUrl = "";
        let snapshotImageHint = "No RGB frame available in snapshot.";
        if (imagePayload.ok) {
          showLiveFrameImage(imagePayload.blob);
          snapshotImageHint = `${imagePayload.blob.type || "image/jpeg"} · ${imagePayload.blob.size} bytes`;
          if (state.captureEnabled) {
            try {
              snapshotImageDataUrl = await blobToDataUrl(imagePayload.blob);
            } catch (_) {
              snapshotImageDataUrl = "";
            }
          }
        } else {
          clearFrameImage("Frame image is not provided by this display.");
        }

        if (state.captureEnabled) {
          pushReplaySnapshot(frame, snapshotImageDataUrl, snapshotImageHint);
        }
      } catch (error) {
        setStatus(`Fetch frame failed: ${error.message}`, "status error");
      }
    }

    async function sendAction(actionValue) {
      if (!state.selectedDisplayId) {
        return;
      }
      if (!state.selectedDisplayAcceptsInput) {
        setStatus("Selected display does not accept input.", "status warn");
        return;
      }
      const body = {
        display_id: state.selectedDisplayId,
        payload: { type: "action", action: actionValue },
        context: {},
      };
      try {
        const response = await apiJson("/ws_rgb/input", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const queued = Number(response.queued || 0);
        setStatus(`Input accepted. queued=${queued}`, queued > 0 ? "status ok" : "status warn");
      } catch (error) {
        setStatus(`Send action failed: ${error.message}`, "status error");
      }
    }

    async function sendKeyEvent(type, key, repeat) {
      if (!state.selectedDisplayId) {
        return;
      }
      if (!state.selectedDisplayAcceptsInput) {
        return;
      }
      const body = {
        display_id: state.selectedDisplayId,
        payload: {
          type,
          key: String(key || ""),
          repeat: Boolean(repeat),
          timestamp_ms: Date.now(),
        },
        context: {},
      };
      try {
        await apiJson("/ws_rgb/input", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
      } catch (error) {
        setStatus(`Send key failed: ${error.message}`, "status error");
      }
    }

    async function pollTick() {
      await refreshDisplays();
      if (state.selectedDisplayReplaySeekable) {
        if (
          !state.replayBufferLoading &&
          state.replayBufferLoadedForDisplay !== state.selectedDisplayId
        ) {
          await preloadReplayBuffer();
        }
        return;
      }
      if (!state.replayMode) {
        await fetchFrame();
      }
    }

    function restartPolling() {
      if (state.pollTimer !== null) {
        clearInterval(state.pollTimer);
      }
      state.pollTimer = setInterval(() => {
        pollTick().catch((error) => {
          setStatus(`Polling failed: ${error.message}`, "status error");
        });
      }, state.pollMs);
    }

    function bindEvents() {
      el.refreshDisplaysBtn.addEventListener("click", () => {
        pollTick();
      });
      el.displaySelect.addEventListener("change", () => {
        const nextDisplayId = el.displaySelect.value;
        if (nextDisplayId !== state.selectedDisplayId) {
          state.selectedDisplayId = nextDisplayId;
          resetReplayBuffer();
          clearFrameImage("Display switched. Waiting frame...");
        }
        updateInputUi();
        if (state.selectedDisplayReplaySeekable) {
          preloadReplayBuffer();
          return;
        }
        fetchFrame();
      });
      el.applyPollBtn.addEventListener("click", () => {
        const parsed = Number(el.pollMsInput.value || "20");
        state.pollMs = Number.isFinite(parsed) ? Math.max(20, Math.min(2000, parsed)) : 20;
        el.pollMsInput.value = String(state.pollMs);
        restartPolling();
        if (state.replayTimer !== null) {
          startReplayPlayback();
        }
        setStatus(`Poll interval set to ${state.pollMs} ms`, "status ok");
      });
      el.fetchFrameBtn.addEventListener("click", () => {
        setReplayMode(false);
        if (state.selectedDisplayReplaySeekable) {
          preloadReplayBuffer();
          return;
        }
        fetchFrame();
      });
      el.sendActionBtn.addEventListener("click", () => {
        sendAction(normalizeAction(el.actionInput.value));
      });
      el.actionInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          sendAction(normalizeAction(el.actionInput.value));
        }
      });
      el.captureToggleBtn.addEventListener("click", () => {
        state.captureEnabled = !state.captureEnabled;
        updateReplayUi();
      });
      el.replayToggleBtn.addEventListener("click", () => {
        if (state.replayTimer === null) {
          startReplayPlayback();
        } else {
          stopReplayPlayback();
          updateReplayUi();
        }
      });
      el.replayStepPrevBtn.addEventListener("click", () => {
        setReplayMode(true);
        stopReplayPlayback();
        renderReplayIndex(state.replayIndex - 1);
      });
      el.replayStepNextBtn.addEventListener("click", () => {
        setReplayMode(true);
        stopReplayPlayback();
        renderReplayIndex(state.replayIndex + 1);
      });
      el.replayLiveBtn.addEventListener("click", () => {
        setReplayMode(false);
        if (state.selectedDisplayReplaySeekable) {
          preloadReplayBuffer();
          return;
        }
        fetchFrame();
      });
      el.replaySpeedSelect.addEventListener("change", () => {
        const parsed = Number(el.replaySpeedSelect.value || "1");
        state.replaySpeed = Number.isFinite(parsed) && parsed > 0 ? parsed : 1.0;
        if (state.replayTimer !== null) {
          startReplayPlayback();
        } else {
          updateReplayUi();
        }
      });
      const applyReplaySeek = () => {
        setReplayMode(true);
        stopReplayPlayback();
        renderReplayIndex(Number(el.replaySeek.value || "0"));
      };
      el.replaySeek.addEventListener("pointerdown", () => {
        state.replaySeekDragging = true;
      });
      el.replaySeek.addEventListener("pointerup", () => {
        state.replaySeekDragging = false;
      });
      el.replaySeek.addEventListener("change", () => {
        state.replaySeekDragging = false;
        applyReplaySeek();
      });
      el.replaySeek.addEventListener("input", applyReplaySeek);
      window.addEventListener("keydown", (event) => {
        if (!el.keyCaptureToggle.checked || !state.selectedDisplayAcceptsInput) {
          return;
        }
        event.preventDefault();
        sendKeyEvent("keydown", event.key, event.repeat);
      });
      window.addEventListener("keyup", (event) => {
        if (!el.keyCaptureToggle.checked || !state.selectedDisplayAcceptsInput) {
          return;
        }
        event.preventDefault();
        sendKeyEvent("keyup", event.key, false);
      });
      window.addEventListener("beforeunload", () => {
        if (state.pollTimer !== null) {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
        }
        stopReplayPlayback();
        releaseLiveFrameImage();
      });
    }

    async function start() {
      const storedLayout = readStoredPanelLayout();
      applyPanelLayout(storedLayout || PANEL_LAYOUT_DEFAULTS);
      updateReplayUi();
      bindEvents();
      bindPanelResizeEvents();
      await refreshDisplays();
      if (state.selectedDisplayReplaySeekable) {
        await preloadReplayBuffer();
      } else {
        await fetchFrame();
      }
      restartPolling();
    }

    start();
  </script>
</body>
</html>"""

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""

        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - match BaseHTTPRequestHandler signature
        """Serve read-only hub endpoints."""

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path in {"/ws_rgb/viewer", "/ws_rgb"}:
            self._send_html(self._VIEWER_HTML, status=HTTPStatus.OK)
            return
        if parsed.path == "/ws_rgb/displays":
            hub = self._hub()
            payload = {
                "ok": True,
                "running": True,
                "base_url": hub.base_url,
                "displays": hub.list_displays(),
            }
            self._send_json(payload, status=HTTPStatus.OK)
            return
        if parsed.path == "/ws_rgb/frame":
            display_id = self._first_param(params, "display_id")
            if not display_id:
                self._send_json({"ok": False, "error": "missing_display_id"}, status=HTTPStatus.BAD_REQUEST)
                return
            replay_index, replay_index_error = self._parse_optional_int_param(params, "replay_index")
            if replay_index_error:
                self._send_json({"ok": False, "error": replay_index_error}, status=HTTPStatus.BAD_REQUEST)
                return
            payload = self._hub().broadcast_frame(display_id, replay_index=replay_index)
            status = self._map_frame_error_status(str(payload.get("error") or "")) if not payload.get("ok") else HTTPStatus.OK
            self._send_json(payload, status=status)
            return
        if parsed.path == "/ws_rgb/replay_buffer":
            display_id = self._first_param(params, "display_id")
            if not display_id:
                self._send_json({"ok": False, "error": "missing_display_id"}, status=HTTPStatus.BAD_REQUEST)
                return
            limit = self._parse_int_param(params, "limit", default=0)
            payload = self._hub().load_replay_frames(display_id, limit=max(0, int(limit)))
            status = self._map_replay_buffer_error_status(str(payload.get("error") or "")) if not payload.get("ok") else HTTPStatus.OK
            self._send_json(payload, status=status)
            return
        if parsed.path in {"/ws_rgb/frame_image", "/ws_rgb/frame.jpg"}:
            display_id = self._first_param(params, "display_id")
            if not display_id:
                self._send_json({"ok": False, "error": "missing_display_id"}, status=HTTPStatus.BAD_REQUEST)
                return
            quality = self._parse_int_param(params, "quality", default=80)
            replay_index, replay_index_error = self._parse_optional_int_param(params, "replay_index")
            if replay_index_error:
                self._send_json({"ok": False, "error": replay_index_error}, status=HTTPStatus.BAD_REQUEST)
                return
            payload = self._hub().broadcast_frame_image(display_id, quality=quality, replay_index=replay_index)
            if payload.get("ok"):
                content_type = str(payload.get("content_type") or "image/jpeg")
                image_bytes = payload.get("image_bytes")
                if not isinstance(image_bytes, (bytes, bytearray)):
                    self._send_json({"ok": False, "error": "frame_image_invalid"}, status=HTTPStatus.UNPROCESSABLE_ENTITY)
                    return
                self._send_binary(
                    bytes(image_bytes),
                    status=HTTPStatus.OK,
                    content_type=content_type,
                )
                return
            error = str(payload.get("error") or "frame_image_unavailable")
            status = self._map_frame_image_error_status(error)
            self._send_json(payload, status=status)
            return
        self._send_json({"ok": False, "error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - match BaseHTTPRequestHandler signature
        """Serve mutable hub endpoints."""

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path != "/ws_rgb/input":
            self._send_json({"ok": False, "error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        body = self._read_json_body()
        if body is None:
            self._send_json({"ok": False, "error": "invalid_json_body"}, status=HTTPStatus.BAD_REQUEST)
            return

        display_id = str(body.get("display_id") or self._first_param(params, "display_id") or "")
        if not display_id:
            self._send_json({"ok": False, "error": "missing_display_id"}, status=HTTPStatus.BAD_REQUEST)
            return
        payload = body.get("payload")
        if not isinstance(payload, Mapping):
            self._send_json({"ok": False, "error": "missing_payload"}, status=HTTPStatus.BAD_REQUEST)
            return
        context = body.get("context")
        if context is not None and not isinstance(context, Mapping):
            self._send_json({"ok": False, "error": "invalid_context"}, status=HTTPStatus.BAD_REQUEST)
            return

        response = self._hub().handle_input(display_id=display_id, payload=payload, context=context)
        status = HTTPStatus.OK if response.get("ok") else HTTPStatus.BAD_REQUEST
        self._send_json(response, status=status)

    def log_message(self, format: str, *args) -> None:  # noqa: A003 - match base signature
        return

    def _hub(self) -> WsRgbHubServer:
        return getattr(self.server, "ws_rgb_hub")

    @staticmethod
    def _first_param(params: Mapping[str, list[str]], key: str) -> Optional[str]:
        values = params.get(key)
        if not values:
            return None
        return values[0]

    @staticmethod
    def _parse_int_param(params: Mapping[str, list[str]], key: str, *, default: int) -> int:
        raw = _WsRgbRequestHandler._first_param(params, key)
        if raw is None:
            return int(default)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _parse_optional_int_param(
        params: Mapping[str, list[str]],
        key: str,
    ) -> tuple[Optional[int], Optional[str]]:
        raw = _WsRgbRequestHandler._first_param(params, key)
        if raw is None:
            return None, None
        try:
            return int(raw), None
        except (TypeError, ValueError):
            return None, f"invalid_{key}"

    @staticmethod
    def _map_frame_image_error_status(error: str) -> HTTPStatus:
        if error in {"display_not_found", "frame_image_missing"}:
            return HTTPStatus.NOT_FOUND
        if error in {"replay_index_unsupported", "invalid_replay_index"}:
            return HTTPStatus.BAD_REQUEST
        if error == "replay_index_fetch_failed":
            return HTTPStatus.INTERNAL_SERVER_ERROR
        if error in {"pillow_missing", "numpy_missing"}:
            return HTTPStatus.SERVICE_UNAVAILABLE
        if error.startswith("jpeg_encode_failed:"):
            return HTTPStatus.INTERNAL_SERVER_ERROR
        if error in {"frame_image_invalid", "frame_image_invalid_shape"}:
            return HTTPStatus.UNPROCESSABLE_ENTITY
        return HTTPStatus.BAD_REQUEST

    @staticmethod
    def _map_frame_error_status(error: str) -> HTTPStatus:
        if error == "display_not_found":
            return HTTPStatus.NOT_FOUND
        if error in {"replay_index_unsupported", "invalid_replay_index"}:
            return HTTPStatus.BAD_REQUEST
        if error == "replay_index_fetch_failed":
            return HTTPStatus.INTERNAL_SERVER_ERROR
        return HTTPStatus.BAD_REQUEST

    @staticmethod
    def _map_replay_buffer_error_status(error: str) -> HTTPStatus:
        if error == "display_not_found":
            return HTTPStatus.NOT_FOUND
        if error in {"replay_buffer_unsupported", "replay_count_failed"}:
            return HTTPStatus.BAD_REQUEST
        if error == "replay_frame_fetch_failed":
            return HTTPStatus.INTERNAL_SERVER_ERROR
        return HTTPStatus.BAD_REQUEST

    def _read_json_body(self) -> Optional[dict[str, Any]]:
        content_length_raw = self.headers.get("Content-Length")
        if not content_length_raw:
            return {}
        try:
            content_length = int(content_length_raw)
        except (TypeError, ValueError):
            return None
        raw = self.rfile.read(max(0, content_length))
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

    def _send_cors_headers(self) -> None:
        allow_origin = str(getattr(self.server, "allow_origin", "*"))
        self.send_header("Access-Control-Allow-Origin", allow_origin)

    def _send_json(self, payload: Mapping[str, Any], *, status: HTTPStatus) -> None:
        body = json.dumps(dict(payload), ensure_ascii=False, default=str).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:  # pragma: no cover - exercised in integration
            if _is_client_disconnect_error(exc):
                return
            raise

    def _send_html(self, html: str, *, status: HTTPStatus) -> None:
        body = str(html).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:  # pragma: no cover - exercised in integration
            if _is_client_disconnect_error(exc):
                return
            raise

    def _send_binary(self, body: bytes, *, status: HTTPStatus, content_type: str) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", str(content_type))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:  # pragma: no cover - exercised in integration
            if _is_client_disconnect_error(exc):
                return
            raise
