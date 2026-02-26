"""In-process RGB hub primitives with optional HTTP control endpoints."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping, Optional
from urllib.parse import parse_qs, urlparse

from gage_eval.role.arena.input_mapping import GameInputMapper, HumanActionEvent


@dataclass
class DisplayRegistration:
    """Display registration metadata for WsRgbHubServer."""

    display_id: str
    label: str
    human_player_id: str
    frame_source: Callable[[], Any]
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
            items.append(
                {
                    "display_id": reg.display_id,
                    "label": reg.label,
                    "human_player_id": reg.human_player_id,
                    "legal_moves": list(reg.legal_moves or []),
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

    def broadcast_frame(self, display_id: str) -> dict[str, Any]:
        """Fetch one latest frame payload from display frame_source."""

        registration = self._displays.get(str(display_id))
        if registration is None:
            return {"ok": False, "error": "display_not_found", "display_id": str(display_id)}
        frame = registration.frame_source()
        return {"ok": True, "display_id": registration.display_id, "frame": frame}

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
            payload = action.to_queue_payload()
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
  <title>GAGE ws_rgb Viewer</title>
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
      padding: 16px;
    }
    .shell {
      max-width: 1200px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    .panel {
      background: linear-gradient(180deg, var(--panel) 0%, var(--panel-alt) 100%);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      box-shadow: 0 6px 22px rgba(0, 0, 0, 0.35);
    }
    .title {
      font-size: 15px;
      font-weight: 700;
      margin-bottom: 10px;
      color: var(--accent);
    }
    .row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .row + .row {
      margin-top: 10px;
    }
    select, input, button {
      background: #0f1727;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 7px 9px;
    }
    select {
      min-width: 300px;
    }
    input {
      min-width: 140px;
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
    .split {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    @media (max-width: 900px) {
      .split {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="panel">
      <div class="title">GAGE ws_rgb Viewer</div>
      <div class="row">
        <label for="displaySelect">Display:</label>
        <select id="displaySelect"></select>
        <button id="refreshDisplaysBtn" type="button">Refresh Displays</button>
      </div>
      <div class="row">
        <label for="pollMsInput">Poll ms:</label>
        <input id="pollMsInput" type="number" min="50" max="2000" step="10" value="200" />
        <button id="applyPollBtn" type="button">Apply</button>
        <button id="fetchFrameBtn" type="button">Fetch Once</button>
      </div>
      <div class="row">
        <label for="actionInput">Action:</label>
        <input id="actionInput" type="text" value="0" />
        <button id="sendActionBtn" type="button">Send Action</button>
        <label><input id="keyCaptureToggle" type="checkbox" /> Key Capture</label>
      </div>
      <div id="status" class="status">Initializing...</div>
    </div>

    <div class="split">
      <div class="panel">
        <div class="title">Frame board_text</div>
        <pre id="boardText">(waiting frame)</pre>
      </div>
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
      pollTimer: null,
      pollMs: 200,
    };

    const el = {
      displaySelect: document.getElementById("displaySelect"),
      refreshDisplaysBtn: document.getElementById("refreshDisplaysBtn"),
      pollMsInput: document.getElementById("pollMsInput"),
      applyPollBtn: document.getElementById("applyPollBtn"),
      fetchFrameBtn: document.getElementById("fetchFrameBtn"),
      actionInput: document.getElementById("actionInput"),
      sendActionBtn: document.getElementById("sendActionBtn"),
      keyCaptureToggle: document.getElementById("keyCaptureToggle"),
      status: document.getElementById("status"),
      boardText: document.getElementById("boardText"),
      frameJson: document.getElementById("frameJson"),
    };

    function setStatus(message, tone = "status") {
      el.status.className = tone;
      el.status.textContent = message;
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
        return;
      }
      const hasOld = state.displays.some(item => String(item.display_id || "") === oldSelected);
      state.selectedDisplayId = hasOld ? oldSelected : String(state.displays[0].display_id || "");
      el.displaySelect.value = state.selectedDisplayId;
    }

    async function refreshDisplays() {
      try {
        const payload = await apiJson("/ws_rgb/displays");
        renderDisplays(payload.displays || []);
        if (!state.selectedDisplayId) {
          setStatus("No display registered yet.", "status warn");
          return;
        }
        setStatus(`Connected. display_id=${state.selectedDisplayId}`, "status ok");
      } catch (error) {
        setStatus(`Refresh displays failed: ${error.message}`, "status error");
      }
    }

    async function fetchFrame() {
      if (!state.selectedDisplayId) {
        return;
      }
      try {
        const query = new URLSearchParams({ display_id: state.selectedDisplayId });
        const payload = await apiJson(`/ws_rgb/frame?${query.toString()}`);
        const frame = payload.frame || {};
        const boardText = frame.board_text != null ? String(frame.board_text) : "(no board_text)";
        el.boardText.textContent = boardText;
        el.frameJson.textContent = JSON.stringify(frame, null, 2);
      } catch (error) {
        setStatus(`Fetch frame failed: ${error.message}`, "status error");
      }
    }

    async function sendAction(actionValue) {
      if (!state.selectedDisplayId) {
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

    async function sendKeyEvent(type, key) {
      if (!state.selectedDisplayId) {
        return;
      }
      const body = {
        display_id: state.selectedDisplayId,
        payload: { type, key: String(key || "") },
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

    function restartPolling() {
      if (state.pollTimer !== null) {
        clearInterval(state.pollTimer);
      }
      state.pollTimer = setInterval(() => {
        refreshDisplays().then(fetchFrame);
      }, state.pollMs);
    }

    function bindEvents() {
      el.refreshDisplaysBtn.addEventListener("click", () => {
        refreshDisplays().then(fetchFrame);
      });
      el.displaySelect.addEventListener("change", () => {
        state.selectedDisplayId = el.displaySelect.value;
        fetchFrame();
      });
      el.applyPollBtn.addEventListener("click", () => {
        const parsed = Number(el.pollMsInput.value || "200");
        state.pollMs = Number.isFinite(parsed) ? Math.max(50, Math.min(2000, parsed)) : 200;
        el.pollMsInput.value = String(state.pollMs);
        restartPolling();
        setStatus(`Poll interval set to ${state.pollMs} ms`, "status ok");
      });
      el.fetchFrameBtn.addEventListener("click", fetchFrame);
      el.sendActionBtn.addEventListener("click", () => {
        sendAction(normalizeAction(el.actionInput.value));
      });
      el.actionInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          sendAction(normalizeAction(el.actionInput.value));
        }
      });
      window.addEventListener("keydown", (event) => {
        if (!el.keyCaptureToggle.checked) {
          return;
        }
        event.preventDefault();
        sendKeyEvent("keydown", event.key);
      });
      window.addEventListener("keyup", (event) => {
        if (!el.keyCaptureToggle.checked) {
          return;
        }
        event.preventDefault();
        sendKeyEvent("keyup", event.key);
      });
    }

    async function start() {
      bindEvents();
      await refreshDisplays();
      await fetchFrame();
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
            payload = self._hub().broadcast_frame(display_id)
            status = HTTPStatus.OK if payload.get("ok") else HTTPStatus.NOT_FOUND
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
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, *, status: HTTPStatus) -> None:
        body = str(html).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)
