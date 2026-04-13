"""HTTP action server for human input in arena games."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue
from threading import Lock, Thread
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from loguru import logger

from gage_eval.role.arena.human_input_protocol import (
    build_action_payload,
    dump_action_payload,
)


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
            action_payload = self._extract_action_payload(parsed)
            if not action_payload.get("action"):
                self._send_json({"error": "Missing action"}, status=HTTPStatus.BAD_REQUEST)
                return

            queue_server: Optional[ActionQueueServer] = getattr(  # type: ignore[attr-defined]
                self.server,
                "action_server_ref",
                None,
            )
            if queue_server is not None and queue_server.has_action_routes():
                if not action_payload.get("sample_id"):
                    self._send_json({"error": "missing_sample_id"}, status=HTTPStatus.BAD_REQUEST)
                    return
                if not action_payload.get("player_id"):
                    self._send_json({"error": "missing_player_id"}, status=HTTPStatus.BAD_REQUEST)
                    return

            queue: Optional[Any] = None
            error: Optional[str] = None
            if queue_server is not None:
                queue, error = queue_server.resolve_action_queue(action_payload.get("sample_id"))
            else:
                queue = getattr(self.server, "action_queue", None)  # type: ignore[attr-defined]
            if queue is None:
                status = HTTPStatus.BAD_REQUEST if error == "missing_sample_id" else HTTPStatus.NOT_FOUND
                if error is None:
                    status = HTTPStatus.INTERNAL_SERVER_ERROR
                    error = "action_queue_not_available"
                self._send_json({"error": error}, status=status)
                return

            queue.put(dump_action_payload(action_payload))
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

    def _extract_action_payload(self, parsed_url) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(length) if length > 0 else b""
        body_text = raw_body.decode("utf-8", errors="ignore").strip()
        if body_text:
            payload = self._try_parse_json(body_text)
            if isinstance(payload, dict):
                return build_action_payload(
                    action=payload.get("action") or payload.get("move"),
                    player_id=(
                        payload.get("player_id")
                        or payload.get("playerId")
                        or payload.get("player")
                        or payload.get("player_idx")
                        or payload.get("playerIdx")
                    ),
                    sample_id=payload.get("sample_id") or payload.get("sampleId"),
                    raw=payload.get("raw"),
                    source="action_server",
                    run_id=payload.get("run_id") or payload.get("runId"),
                    task_id=payload.get("task_id") or payload.get("taskId"),
                    display_id=payload.get("display_id") or payload.get("displayId"),
                    chat=payload.get("chat"),
                    metadata=payload.get("metadata"),
                )
            return build_action_payload(
                action=body_text,
                raw=body_text,
                source="action_server",
            )

        params = parse_qs(parsed_url.query)
        return build_action_payload(
            action=_first_param(params, "action") or _first_param(params, "move"),
            player_id=(
                _first_param(params, "player_id")
                or _first_param(params, "playerId")
                or _first_param(params, "player_idx")
                or _first_param(params, "playerIdx")
            ),
            sample_id=_first_param(params, "sample_id") or _first_param(params, "sampleId"),
            source="action_server",
        )

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
        self._action_routes: dict[str, Any] = {}
        self._route_lock = Lock()
        self._server = HTTPServer((self._host, self._port), ActionRequestHandler)
        setattr(self._server, "action_queue", self._queue)
        setattr(self._server, "chat_queue", self._chat_queue)
        setattr(self._server, "allow_origin", self._allow_origin)
        setattr(self._server, "action_server_ref", self)
        self._thread: Optional[Thread] = None

    @property
    def action_queue(self) -> Queue[str]:
        """Return the queue that buffers incoming actions."""

        return self._queue

    @property
    def chat_queue(self) -> Queue[dict[str, str]]:
        """Return the queue that buffers incoming chat messages."""

        return self._chat_queue

    def register_action_queue(self, sample_id: str, action_queue: Any) -> None:
        """Register a sample-scoped queue or router for action delivery."""

        normalized_sample_id = str(sample_id)
        with self._route_lock:
            self._action_routes[normalized_sample_id] = action_queue

    def unregister_action_queue(self, sample_id: str) -> None:
        """Remove a sample-scoped action route."""

        normalized_sample_id = str(sample_id)
        with self._route_lock:
            self._action_routes.pop(normalized_sample_id, None)

    def resolve_action_queue(
        self,
        sample_id: Optional[str],
    ) -> tuple[Optional[Any], Optional[str]]:
        """Resolve the target queue for one incoming action payload."""

        with self._route_lock:
            if self._action_routes:
                if sample_id is None:
                    return None, "missing_sample_id"
                queue = self._action_routes.get(str(sample_id))
                if queue is None:
                    return None, "sample_route_not_found"
                return queue, None
        return self._queue, None

    def submit_action_payload(self, payload: dict[str, Any]) -> Optional[str]:
        """Route one normalized action payload into the matching queue."""

        queue, error = self.resolve_action_queue(payload.get("sample_id"))
        if queue is None:
            return error or "action_queue_not_available"
        queue.put(dump_action_payload(payload))
        return None

    def has_action_routes(self) -> bool:
        """Return whether sample-scoped action routes are registered."""

        with self._route_lock:
            return bool(self._action_routes)

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
