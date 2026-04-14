"""HTTP action server for human input in arena games."""

from __future__ import annotations

import json
from email.message import Message
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from queue import Queue
from threading import Lock, Thread
from typing import Any, Optional
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import BaseHandler, build_opener, install_opener
from urllib.response import addinfourl

from loguru import logger

from gage_eval.role.arena.human_input_protocol import (
    build_action_payload,
    dump_action_payload,
)

_IN_PROCESS_ACTION_SERVERS: dict[tuple[str, int], Any] = {}
_IN_PROCESS_ACTION_SERVERS_LOCK = Lock()
_IN_PROCESS_NEXT_PORT = 39000
_IN_PROCESS_OPENER_INSTALLED = False


def _first_param(params: dict[str, list[str]], key: str) -> Optional[str]:
    values = params.get(key)
    if not values:
        return None
    return values[0]


def _try_parse_json(text: str) -> Optional[dict[str, Any]]:
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_action_payload(path: str, body: bytes) -> dict[str, Any]:
    parsed_url = urlparse(path)
    body_text = body.decode("utf-8", errors="ignore").strip()
    if body_text:
        payload = _try_parse_json(body_text)
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


def _extract_chat(path: str, body: bytes) -> tuple[str, Optional[str]]:
    parsed_url = urlparse(path)
    body_text = body.decode("utf-8", errors="ignore").strip()
    if body_text:
        payload = _try_parse_json(body_text)
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


def _dispatch_action_server_request(
    server: Any,
    *,
    method: str,
    path: str,
    body: bytes,
) -> tuple[HTTPStatus, dict[str, object]]:
    if method == "OPTIONS":
        return HTTPStatus.NO_CONTENT, {}

    parsed = urlparse(path)
    if method == "POST" and parsed.path == "/tournament/action":
        action_payload = _extract_action_payload(path, body)
        if not action_payload.get("action"):
            return HTTPStatus.BAD_REQUEST, {"error": "Missing action"}

        queue_server: Optional[ActionQueueServer] = getattr(
            server,
            "action_server_ref",
            None,
        )
        if queue_server is not None and queue_server.has_action_routes():
            if not action_payload.get("sample_id"):
                return HTTPStatus.BAD_REQUEST, {"error": "missing_sample_id"}
            if not action_payload.get("player_id"):
                return HTTPStatus.BAD_REQUEST, {"error": "missing_player_id"}

        queue: Optional[Any] = None
        error: Optional[str] = None
        if queue_server is not None:
            queue, error = queue_server.resolve_action_queue(action_payload.get("sample_id"))
        else:
            queue = getattr(server, "action_queue", None)
        if queue is None:
            status = HTTPStatus.BAD_REQUEST if error == "missing_sample_id" else HTTPStatus.NOT_FOUND
            if error is None:
                status = HTTPStatus.INTERNAL_SERVER_ERROR
                error = "action_queue_not_available"
            return status, {"error": error}

        queue.put(dump_action_payload(action_payload))
        return HTTPStatus.OK, {"status": "queued"}

    if method == "POST" and parsed.path == "/tournament/chat":
        chat_text, player_id = _extract_chat(path, body)
        if not chat_text:
            return HTTPStatus.BAD_REQUEST, {"error": "Missing chat"}

        chat_queue: Optional[Queue[dict[str, str]]] = getattr(
            server,
            "chat_queue",
            None,
        )
        if chat_queue is None:
            return HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Chat queue not available"}

        payload = {"text": str(chat_text)}
        if player_id:
            payload["player_id"] = str(player_id)
        chat_queue.put(payload)
        return HTTPStatus.OK, {"status": "queued"}

    return HTTPStatus.NOT_FOUND, {"error": "Not Found"}


def _build_response_headers(*, allow_origin: str, content_length: int) -> Message:
    headers = Message()
    headers.add_header("Content-Type", "application/json; charset=utf-8")
    headers.add_header("Content-Length", str(content_length))
    headers.add_header("Access-Control-Allow-Origin", allow_origin)
    headers.add_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    headers.add_header("Access-Control-Allow-Headers", "Content-Type")
    return headers


def _allocate_in_process_port() -> int:
    global _IN_PROCESS_NEXT_PORT

    with _IN_PROCESS_ACTION_SERVERS_LOCK:
        port = _IN_PROCESS_NEXT_PORT
        _IN_PROCESS_NEXT_PORT += 1
    return port


class _InProcessHTTPServer:
    """Emulate a minimal HTTP server when socket binding is unavailable."""

    def __init__(self, host: str, port: int) -> None:
        """Initialize the in-process server transport."""

        self.server_address = (host, port)
        self.action_queue: Any = None
        self.chat_queue: Any = None
        self.allow_origin = "*"
        self.action_server_ref: Any = None
        with _IN_PROCESS_ACTION_SERVERS_LOCK:
            _IN_PROCESS_ACTION_SERVERS[(host, port)] = self

    def serve_forever(self) -> None:
        """Mirror HTTPServer's serve_forever interface."""

    def shutdown(self) -> None:
        """Unregister the in-process server from the transport registry."""

        self.server_close()

    def server_close(self) -> None:
        """Remove the in-process server from the transport registry."""

        host, port = self.server_address[:2]
        with _IN_PROCESS_ACTION_SERVERS_LOCK:
            _IN_PROCESS_ACTION_SERVERS.pop((str(host), int(port)), None)


class _InProcessActionServerHandler(BaseHandler):
    """Intercept loopback requests for in-process action servers."""

    handler_order = 100

    def http_open(self, request):
        """Serve matching requests through the in-process transport."""

        parsed = urlparse(request.full_url)
        host = str(parsed.hostname or "")
        port = int(parsed.port or 80)
        with _IN_PROCESS_ACTION_SERVERS_LOCK:
            server = _IN_PROCESS_ACTION_SERVERS.get((host, port))
        if server is None:
            return None

        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        status, payload = _dispatch_action_server_request(
            server,
            method=request.get_method(),
            path=path,
            body=request.data or b"",
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = _build_response_headers(
            allow_origin=str(getattr(server, "allow_origin", "*")),
            content_length=len(body),
        )
        response = addinfourl(BytesIO(body), headers, request.full_url, code=int(status))
        response.msg = status.phrase
        if int(status) >= 400:
            raise HTTPError(request.full_url, int(status), status.phrase, headers, response)
        return response


def _ensure_in_process_opener_installed() -> None:
    global _IN_PROCESS_OPENER_INSTALLED

    with _IN_PROCESS_ACTION_SERVERS_LOCK:
        if _IN_PROCESS_OPENER_INSTALLED:
            return
        install_opener(build_opener(_InProcessActionServerHandler))
        _IN_PROCESS_OPENER_INSTALLED = True


def _configure_server(server: Any, *, action_server: "ActionQueueServer") -> None:
    setattr(server, "action_queue", action_server._queue)
    setattr(server, "chat_queue", action_server._chat_queue)
    setattr(server, "allow_origin", action_server._allow_origin)
    setattr(server, "action_server_ref", action_server)


class ActionRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler that accepts human actions and enqueues them."""

    server_version = "GAGEActionServer/1.0"

    def do_OPTIONS(self) -> None:
        """Reply with CORS headers for preflight requests."""

        self._send_dispatch_response(
            *_dispatch_action_server_request(
                self.server,
                method="OPTIONS",
                path=self.path,
                body=b"",
            )
        )

    def do_POST(self) -> None:  # noqa: N802 - match BaseHTTPRequestHandler signature
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        self._send_dispatch_response(
            *_dispatch_action_server_request(
                self.server,
                method="POST",
                path=self.path,
                body=body,
            )
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - match base signature
        logger.debug("ActionServer {}", format % args)

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

    def _send_dispatch_response(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        if status == HTTPStatus.NO_CONTENT:
            self.send_response(status)
            self._send_cors_headers()
            self.end_headers()
            return
        self._send_json(payload, status=status)


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
        try:
            self._server = HTTPServer((self._host, self._port), ActionRequestHandler)
        except PermissionError:
            fallback_port = self._port if self._port > 0 else _allocate_in_process_port()
            self._server = _InProcessHTTPServer(self._host, fallback_port)
            _ensure_in_process_opener_installed()
            logger.warning(
                "Socket bind is unavailable for {}:{}, using in-process action server transport.",
                self._host,
                self._port,
            )
        _configure_server(self._server, action_server=self)
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
        host, port = self._server.server_address[:2]
        logger.info("Action server listening on http://{}:{}", host, port)

    def stop(self) -> None:
        """Stop the HTTP server."""

        self._server.shutdown()
        self._server.server_close()
        if self._thread:
            self._thread.join(timeout=2)


__all__ = ["ActionQueueServer"]
