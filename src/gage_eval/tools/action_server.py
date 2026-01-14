"""HTTP action server for human input in arena games."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue
from threading import Thread
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from loguru import logger


def _first_param(params: dict[str, list[str]], key: str) -> Optional[str]:
    values = params.get(key)
    if not values:
        return None
    return values[0]


class ActionRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler that accepts human actions and enqueues them."""

    server_version = "GAGEActionServer/1.0"

    def do_OPTIONS(self) -> None:
        """Reply with CORS headers for preflight requests."""

        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - match BaseHTTPRequestHandler signature
        parsed = urlparse(self.path)
        if parsed.path == "/tournament/action":
            action_text = self._extract_action(parsed)
            if not action_text:
                self._send_json({"error": "Missing action"}, status=HTTPStatus.BAD_REQUEST)
                return

            queue: Optional[Queue[str]] = getattr(self.server, "action_queue", None)  # type: ignore[attr-defined]
            if queue is None:
                self._send_json({"error": "Action queue not available"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            queue.put(action_text)
            self._send_json({"status": "queued"}, status=HTTPStatus.OK)
            return

        if parsed.path == "/tournament/chat":
            chat_text, player_id = self._extract_chat(parsed)
            if not chat_text:
                self._send_json({"error": "Missing chat"}, status=HTTPStatus.BAD_REQUEST)
                return

            chat_queue: Optional[Queue[dict[str, str]]] = getattr(  # type: ignore[attr-defined]
                self.server,
                "chat_queue",
                None,
            )
            if chat_queue is None:
                self._send_json({"error": "Chat queue not available"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            payload = {"text": str(chat_text)}
            if player_id:
                payload["player_id"] = str(player_id)
            chat_queue.put(payload)
            self._send_json({"status": "queued"}, status=HTTPStatus.OK)
            return

        self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - match base signature
        logger.debug("ActionServer {}", format % args)

    def _extract_action(self, parsed_url) -> str:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(length) if length > 0 else b""
        body_text = raw_body.decode("utf-8", errors="ignore").strip()
        if body_text:
            payload = self._try_parse_json(body_text)
            if isinstance(payload, dict):
                action = payload.get("action") or payload.get("move")
                chat = payload.get("chat")
                if action:
                    if chat:
                        return json.dumps(
                            {"action": str(action), "chat": str(chat)},
                            ensure_ascii=False,
                        )
                    return str(action)
            if body_text:
                return body_text

        params = parse_qs(parsed_url.query)
        action = _first_param(params, "action") or _first_param(params, "move")
        return str(action) if action else ""

    def _extract_chat(self, parsed_url) -> tuple[str, Optional[str]]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(length) if length > 0 else b""
        body_text = raw_body.decode("utf-8", errors="ignore").strip()
        if body_text:
            payload = self._try_parse_json(body_text)
            if isinstance(payload, dict):
                chat = payload.get("chat") or payload.get("text") or payload.get("message")
                player_id = (
                    payload.get("player_id")
                    or payload.get("playerId")
                    or payload.get("player")
                    or payload.get("player_idx")
                    or payload.get("playerIdx")
                )
                return (str(chat).strip() if chat else "", str(player_id) if player_id else None)
            return (body_text, None)

        params = parse_qs(parsed_url.query)
        chat = (
            _first_param(params, "chat")
            or _first_param(params, "text")
            or _first_param(params, "message")
        )
        player_id = (
            _first_param(params, "player_id")
            or _first_param(params, "playerId")
            or _first_param(params, "player_idx")
            or _first_param(params, "playerIdx")
        )
        return (str(chat).strip() if chat else "", str(player_id) if player_id else None)

    def _try_parse_json(self, text: str) -> Optional[dict[str, Any]]:
        if not text.startswith("{"):
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _send_cors_headers(self) -> None:
        allow_origin = getattr(self.server, "allow_origin", "*")  # type: ignore[attr-defined]
        self.send_header("Access-Control-Allow-Origin", str(allow_origin))
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, payload: dict[str, object], *, status: HTTPStatus) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)


class ActionQueueServer:
    """Serve an HTTP endpoint that buffers actions into a queue."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8001, *, allow_origin: str = "*") -> None:
        """Initialize the action server.

        Args:
            host: Host interface to bind.
            port: Port to listen on.
            allow_origin: CORS allow-origin header value.
        """

        self._host = str(host)
        self._port = int(port)
        self._allow_origin = str(allow_origin)
        self._queue: Queue[str] = Queue()
        self._chat_queue: Queue[dict[str, str]] = Queue()
        self._server = HTTPServer((self._host, self._port), ActionRequestHandler)
        setattr(self._server, "action_queue", self._queue)
        setattr(self._server, "chat_queue", self._chat_queue)
        setattr(self._server, "allow_origin", self._allow_origin)
        self._thread: Optional[Thread] = None

    @property
    def action_queue(self) -> Queue[str]:
        """Return the queue that buffers incoming actions."""

        return self._queue

    @property
    def chat_queue(self) -> Queue[dict[str, str]]:
        """Return the queue that buffers incoming chat messages."""

        return self._chat_queue

    def start(self) -> None:
        """Start the HTTP server in a background thread."""

        if self._thread and self._thread.is_alive():
            return
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("Action server listening on http://{}:{}", self._host, self._port)

    def stop(self) -> None:
        """Stop the HTTP server."""

        self._server.shutdown()
        self._server.server_close()
        if self._thread:
            self._thread.join(timeout=2)


__all__ = ["ActionQueueServer"]
