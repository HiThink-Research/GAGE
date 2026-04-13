from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from mimetypes import guess_type
from pathlib import Path
import time
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, quote, unquote, unquote_to_bytes, urlparse

from loguru import logger

from gage_eval.role.arena.visualization.contracts import ActionIntentReceipt, ObserverRef
from gage_eval.role.arena.visualization.gateway_service import (
    ArenaVisualGatewayQueryService,
    resolve_registered_visualization_spec,
)
from gage_eval.role.arena.visualization.live_session import (
    ArenaVisualLiveRegistry,
    ArenaVisualLiveSessionSource,
)
from gage_eval.role.arena.visualization.artifacts import to_scene_json_safe

ManifestResolver = Callable[[str, str | None], str | Path | None]
ActionSubmitter = Callable[[str, str | None, dict[str, Any]], ActionIntentReceipt]
ChatSubmitter = Callable[[str, str | None, dict[str, Any]], ActionIntentReceipt]
ControlSubmitter = Callable[[str, str | None, dict[str, Any]], ActionIntentReceipt]
_SESSION_MANIFEST_SUFFIX = Path("arena_visual_session") / "v1" / "manifest.json"
_LOW_LATENCY_STREAM_POLL_INTERVAL_S = 1.0 / 180.0
_LIVE_UPDATE_STREAM_WAIT_TIMEOUT_S = 1.0
_WEBSOCKET_ACCEPT_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_WS_OPCODE_TEXT = 0x1
_WS_OPCODE_CLOSE = 0x8
_WS_OPCODE_PING = 0x9
_WS_OPCODE_PONG = 0xA


class SessionResolutionError(Exception):
    def __init__(
        self,
        *,
        error_code: str,
        status: HTTPStatus,
        session_id: str | None = None,
    ) -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.status = status
        self.session_id = session_id


def _first_param(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key)
    if not values:
        return None
    return values[0]


def _parse_optional_int(value: str | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_{field_name}") from exc


def _parse_observer_override(params: dict[str, list[str]]) -> ObserverRef | None:
    observer_kind = _first_param(params, "observer_kind")
    observer_id = _first_param(params, "observer_id")
    normalized_kind = None if observer_kind is None else str(observer_kind).strip()
    normalized_id = "" if observer_id is None else str(observer_id).strip()

    if normalized_kind is None and normalized_id == "":
        return None
    if not normalized_kind:
        raise ValueError("invalid_observer_kind")

    try:
        observer = ObserverRef(
            observer_id=normalized_id,
            observer_kind=normalized_kind,
        )
    except ValueError as exc:
        raise ValueError("invalid_observer_kind") from exc

    if observer.observer_kind == "player" and observer.observer_id == "":
        raise ValueError("missing_observer_id")
    return observer


def _validate_lookup_token(value: str | None, *, error_code: str) -> str:
    if value is None or value == "":
        raise SessionResolutionError(error_code=error_code, status=HTTPStatus.BAD_REQUEST)
    if value in {".", ".."}:
        raise SessionResolutionError(error_code=error_code, status=HTTPStatus.BAD_REQUEST)
    if "/" in value or "\\" in value or "\x00" in value:
        raise SessionResolutionError(error_code=error_code, status=HTTPStatus.BAD_REQUEST)
    return value


def build_session_manifest_resolver(base_dir: str | Path) -> ManifestResolver:
    root = Path(base_dir).expanduser().resolve()
    cache: dict[tuple[str, str | None], Path] = {}

    def _resolve_candidate(candidate: Path) -> Path | None:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            return None
        if root not in resolved.parents and resolved != root:
            return None
        return resolved if resolved.exists() else None

    def _candidate_bases() -> tuple[Path, ...]:
        candidates = [root]
        if root.name != "runs":
            candidates.append(root / "runs")
        unique: list[Path] = []
        for candidate in candidates:
            if candidate not in unique:
                unique.append(candidate)
        return tuple(unique)

    def _resolver(session_id: str, run_id: str | None = None) -> Path | None:
        normalized_session_id = _validate_lookup_token(
            None if session_id is None else str(session_id),
            error_code="invalid_session_id",
        )
        normalized_run_id = None
        if run_id is not None:
            normalized_run_id = _validate_lookup_token(str(run_id), error_code="invalid_run_id")
        cache_key = (normalized_session_id, normalized_run_id)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        if normalized_run_id is not None:
            matches = [
                resolved
                for resolved in (
                    _resolve_candidate(base / normalized_run_id / "replays" / normalized_session_id / _SESSION_MANIFEST_SUFFIX)
                    for base in _candidate_bases()
                )
                if resolved is not None
            ]
        else:
            direct_candidates = [
                candidate
                for base in _candidate_bases()
                for candidate in (
                    base / normalized_session_id / _SESSION_MANIFEST_SUFFIX,
                    base / "replays" / normalized_session_id / _SESSION_MANIFEST_SUFFIX,
                )
            ]
            matches = [
                resolved
                for resolved in (_resolve_candidate(candidate) for candidate in direct_candidates)
                if resolved is not None
            ]
            if not matches:
                literal_candidates: list[Path] = []
                for base in _candidate_bases():
                    if not base.exists():
                        continue
                    for run_dir in base.iterdir():
                        if not run_dir.is_dir():
                            continue
                        literal_candidates.append(
                            run_dir / "replays" / normalized_session_id / _SESSION_MANIFEST_SUFFIX
                        )
                matches = [
                    resolved
                    for resolved in (_resolve_candidate(candidate) for candidate in literal_candidates)
                    if resolved is not None
                ]

        unique_matches = tuple(dict.fromkeys(matches))
        if len(unique_matches) > 1:
            raise SessionResolutionError(
                error_code="session_ambiguous",
                status=HTTPStatus.CONFLICT,
                session_id=normalized_session_id,
            )
        resolved = unique_matches[0] if unique_matches else None
        if resolved is not None and normalized_run_id is not None:
            cache[cache_key] = resolved
        return resolved

    return _resolver


class ArenaVisualRequestHandler(BaseHTTPRequestHandler):
    server_version = "GAGEArenaVisualServer/1.0"

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - match BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/arena_visual/"):
            self._handle_app_request(parsed.path)
            return
        route = self._match_session_route(parsed.path)
        if route is None:
            self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)
            return
        session_id, suffix = route
        run_id = self._query_run_id(parsed)
        if self._validate_route_tokens(session_id, run_id=run_id) is None:
            return

        if not suffix:
            self._handle_session(session_id, parsed, run_id=run_id)
            return
        if suffix == ("timeline",):
            self._handle_timeline(session_id, parsed, run_id=run_id)
            return
        if suffix == ("events",):
            self._handle_session_updates(session_id, parsed, run_id=run_id)
            return
        if suffix == ("scene",):
            self._handle_scene(session_id, parsed, run_id=run_id)
            return
        if suffix == ("actions", "ws"):
            self._handle_action_websocket(session_id, run_id=run_id)
            return
        if suffix == ("markers",):
            self._handle_markers(session_id, parsed, run_id=run_id)
            return
        if len(suffix) == 2 and suffix[0] == "media":
            self._handle_media(session_id, parsed, media_id=suffix[1], run_id=run_id)
            return
        if len(suffix) == 3 and suffix[0] == "media" and suffix[2] == "stream":
            self._handle_media_stream(session_id, media_id=suffix[1], run_id=run_id)
            return

        self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - match BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        route = self._match_session_route(parsed.path)
        if route is None:
            self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)
            return
        session_id, suffix = route
        run_id = self._query_run_id(parsed)
        if self._validate_route_tokens(session_id, run_id=run_id) is None:
            return
        if suffix == ("actions",):
            self._handle_submit_action(
                session_id,
                run_id=run_id,
                fast_ack=self._query_fast_ack(parsed),
            )
            return
        if suffix == ("chat",):
            self._handle_submit_chat(session_id, run_id=run_id)
            return
        if suffix == ("control",):
            self._handle_submit_control(session_id, run_id=run_id)
            return
        self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - match base signature
        message = format % args
        if _should_suppress_access_log(
            command=getattr(self, "command", None),
            path=getattr(self, "path", None),
            message=message,
        ):
            return
        logger.debug("ArenaVisualServer {}", message)

    def _handle_session(self, session_id: str, parsed_url, *, run_id: str | None) -> None:
        params = parse_qs(parsed_url.query)
        try:
            observer = _parse_observer_override(params)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        live_source = self._app.resolve_live_session(session_id, run_id=run_id)
        if live_source is not None:
            try:
                session = live_source.load_session(observer=observer)
            except Exception:
                logger.exception("Failed to load live visual session {}", session_id)
                self._send_json({"error": "session_load_failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(session.to_dict(), status=HTTPStatus.OK)
            return
        manifest_path = self._require_manifest_path(session_id, run_id=run_id)
        if manifest_path is None:
            return
        try:
            session = self._app.query_service.load_session(manifest_path, observer=observer)
        except Exception:
            logger.exception("Failed to load visual session {}", session_id)
            self._send_json({"error": "session_load_failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(session.to_dict(), status=HTTPStatus.OK)

    def _handle_timeline(self, session_id: str, parsed_url, *, run_id: str | None) -> None:
        params = parse_qs(parsed_url.query)
        try:
            after_seq = _parse_optional_int(_first_param(params, "after_seq"), field_name="after_seq")
            limit = _parse_optional_int(_first_param(params, "limit"), field_name="limit")
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if after_seq is not None and after_seq < 0:
            self._send_json({"error": "invalid_after_seq"}, status=HTTPStatus.BAD_REQUEST)
            return
        if limit is not None and limit <= 0:
            self._send_json({"error": "invalid_limit"}, status=HTTPStatus.BAD_REQUEST)
            return
        live_source = self._app.resolve_live_session(session_id, run_id=run_id)
        if live_source is not None:
            try:
                page = live_source.page_timeline(
                    after_seq=after_seq,
                    limit=50 if limit is None else limit,
                )
            except Exception:
                logger.exception("Failed to page live timeline for {}", session_id)
                self._send_json({"error": "timeline_load_failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            payload = {
                "sessionId": session_id,
                "afterSeq": page.after_seq,
                "nextAfterSeq": page.next_after_seq,
                "limit": page.limit,
                "hasMore": page.has_more,
                "events": [event.to_dict() for event in page.events],
            }
            self._send_json(payload, status=HTTPStatus.OK)
            return
        manifest_path = self._require_manifest_path(session_id, run_id=run_id)
        if manifest_path is None:
            return

        try:
            page = self._app.query_service.page_timeline(
                manifest_path,
                after_seq=after_seq,
                limit=50 if limit is None else limit,
            )
        except Exception:
            logger.exception("Failed to page timeline for {}", session_id)
            self._send_json({"error": "timeline_load_failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        payload = {
            "sessionId": session_id,
            "afterSeq": page.after_seq,
            "nextAfterSeq": page.next_after_seq,
            "limit": page.limit,
            "hasMore": page.has_more,
            "events": [event.to_dict() for event in page.events],
        }
        self._send_json(payload, status=HTTPStatus.OK)

    def _handle_session_updates(self, session_id: str, parsed_url, *, run_id: str | None) -> None:
        params = parse_qs(parsed_url.query)
        try:
            observer = _parse_observer_override(params)
            after_seq = _parse_optional_int(_first_param(params, "after_seq"), field_name="after_seq")
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if after_seq is not None and after_seq < 0:
            self._send_json({"error": "invalid_after_seq"}, status=HTTPStatus.BAD_REQUEST)
            return
        live_source = self._app.resolve_live_session(session_id, run_id=run_id)
        if live_source is None:
            self._send_json({"error": "live_updates_not_available"}, status=HTTPStatus.NOT_FOUND)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self._send_cors_headers()
        self.end_headers()

        current_after_seq = after_seq
        last_loaded_revision: int | None = None
        last_session_payload: str | None = None
        last_safe_session_payload: dict[str, object] | None = None
        last_session_lifecycle = "live_running"
        timeline_backlog_pending = True
        while True:
            current_revision = live_source.current_live_revision()
            revision_changed = (
                last_loaded_revision is None or current_revision != last_loaded_revision
            )
            safe_session_payload = last_safe_session_payload
            serialized_session_payload = last_session_payload
            page = None
            try:
                if revision_changed:
                    session = live_source.load_session(observer=observer)
                    session_payload = session.to_dict()
                    safe_session_payload = to_scene_json_safe(session_payload)
                    serialized_session_payload = json.dumps(
                        safe_session_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    last_session_lifecycle = session.lifecycle
                    last_loaded_revision = current_revision
                if revision_changed or timeline_backlog_pending:
                    page = live_source.page_timeline(after_seq=current_after_seq, limit=50)
            except Exception:
                logger.exception("Failed to stream live session updates for {}", session_id)
                break
            timeline_payload = None
            has_session_delta = (
                revision_changed and serialized_session_payload != last_session_payload
            )
            has_timeline_delta = False
            if page is not None:
                timeline_payload = {
                    "sessionId": session_id,
                    "afterSeq": page.after_seq,
                    "nextAfterSeq": page.next_after_seq,
                    "limit": page.limit,
                    "hasMore": page.has_more,
                    "events": [event.to_dict() for event in page.events],
                }
                has_timeline_delta = bool(page.events)
                timeline_backlog_pending = bool(page.has_more)
            else:
                timeline_backlog_pending = False
            if has_session_delta or has_timeline_delta:
                if page is not None and page.next_after_seq is not None:
                    current_after_seq = page.next_after_seq
                payload = {
                    "session": safe_session_payload,
                    "timeline": timeline_payload,
                }
                try:
                    self._send_sse_event("delta", payload)
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    break
                if has_session_delta and serialized_session_payload is not None:
                    last_session_payload = serialized_session_payload
                    last_safe_session_payload = safe_session_payload
            else:
                try:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    break
            if timeline_backlog_pending:
                continue

            try:
                next_revision = live_source.wait_for_live_revision(
                    after_revision=current_revision,
                    timeout_s=_LIVE_UPDATE_STREAM_WAIT_TIMEOUT_S,
                )
            except Exception:
                break
            if next_revision <= current_revision and last_session_lifecycle != "live_running":
                break

    def _handle_scene(self, session_id: str, parsed_url, *, run_id: str | None) -> None:
        params = parse_qs(parsed_url.query)
        try:
            seq = _parse_optional_int(_first_param(params, "seq"), field_name="seq")
            observer = _parse_observer_override(params)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if seq is None:
            self._send_json({"error": "missing_seq"}, status=HTTPStatus.BAD_REQUEST)
            return
        if seq < 0:
            self._send_json({"error": "invalid_seq"}, status=HTTPStatus.BAD_REQUEST)
            return
        live_source = self._app.resolve_live_session(session_id, run_id=run_id)
        if live_source is not None:
            try:
                scene = live_source.load_scene(seq=seq, observer=observer)
            except Exception:
                logger.exception("Failed to load live scene {} for {}", seq, session_id)
                self._send_json({"error": "scene_load_failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if scene is None:
                self._send_json({"error": "scene_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(scene.to_dict(), status=HTTPStatus.OK)
            return
        manifest_path = self._require_manifest_path(session_id, run_id=run_id)
        if manifest_path is None:
            return

        try:
            scene = self._app.query_service.load_scene(manifest_path, seq=seq, observer=observer)
        except Exception:
            logger.exception("Failed to load scene {} for {}", seq, session_id)
            self._send_json({"error": "scene_load_failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if scene is None:
            self._send_json({"error": "scene_not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json(scene.to_dict(), status=HTTPStatus.OK)

    def _handle_markers(self, session_id: str, parsed_url, *, run_id: str | None) -> None:
        params = parse_qs(parsed_url.query)
        marker = str(_first_param(params, "marker") or "").strip()
        if not marker:
            self._send_json({"error": "missing_marker"}, status=HTTPStatus.BAD_REQUEST)
            return
        live_source = self._app.resolve_live_session(session_id, run_id=run_id)
        if live_source is not None:
            try:
                seqs = live_source.lookup_marker(marker)
            except Exception:
                logger.exception("Failed to lookup live marker {} for {}", marker, session_id)
                self._send_json({"error": "marker_lookup_failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(
                {
                    "sessionId": session_id,
                    "marker": marker,
                    "seqs": list(seqs),
                },
                status=HTTPStatus.OK,
            )
            return
        manifest_path = self._require_manifest_path(session_id, run_id=run_id)
        if manifest_path is None:
            return
        try:
            seqs = self._app.query_service.lookup_marker(manifest_path, marker)
        except Exception:
            logger.exception("Failed to lookup marker {} for {}", marker, session_id)
            self._send_json({"error": "marker_lookup_failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(
            {
                "sessionId": session_id,
                "marker": marker,
                "seqs": list(seqs),
            },
            status=HTTPStatus.OK,
        )

    def _handle_media(self, session_id: str, parsed_url, *, media_id: str, run_id: str | None) -> None:
        params = parse_qs(parsed_url.query)
        serve_content = _first_param(params, "content") in {"1", "true", "yes"}
        live_source = self._app.resolve_live_session(session_id, run_id=run_id)
        if live_source is not None:
            try:
                media = live_source.lookup_media(media_id)
            except Exception:
                logger.exception("Failed to lookup live media {} for {}", media_id, session_id)
                self._send_json({"error": "media_lookup_failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if media is None:
                self._send_json({"error": "media_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            if serve_content:
                try:
                    content_payload = live_source.load_media_content(media_id)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                except Exception:
                    logger.exception("Failed to serve live media {} for {}", media_id, session_id)
                    self._send_json({"error": "media_lookup_failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                if content_payload is None:
                    self._send_json({"error": "media_content_not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                content, mime_type = content_payload
                self._send_bytes(
                    content,
                    status=HTTPStatus.OK,
                    mime_type=mime_type or media.mime_type or "application/octet-stream",
                )
                return
            self._send_json(media.to_dict(), status=HTTPStatus.OK)
            return
        manifest_path = self._require_manifest_path(session_id, run_id=run_id)
        if manifest_path is None:
            return
        try:
            media = self._app.query_service.lookup_media(manifest_path, media_id)
        except Exception:
            logger.exception("Failed to lookup media {} for {}", media_id, session_id)
            self._send_json({"error": "media_lookup_failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if media is None:
            self._send_json({"error": "media_not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        if serve_content:
            try:
                content, mime_type = _load_media_content(manifest_path=manifest_path, media_url=media.url)
            except FileNotFoundError:
                self._send_json({"error": "media_content_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_bytes(
                content,
                status=HTTPStatus.OK,
                mime_type=media.mime_type or mime_type or "application/octet-stream",
            )
            return
        self._send_json(media.to_dict(), status=HTTPStatus.OK)

    def _handle_media_stream(self, session_id: str, *, media_id: str, run_id: str | None) -> None:
        live_source = self._app.resolve_live_session(session_id, run_id=run_id)
        if live_source is None:
            self._send_json({"error": "media_stream_not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            media = live_source.lookup_media(media_id)
        except Exception:
            logger.exception("Failed to lookup live media stream {} for {}", media_id, session_id)
            self._send_json({"error": "media_lookup_failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if media is None or media.transport != "low_latency_channel":
            self._send_json({"error": "media_stream_not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._stream_low_latency_media(live_source=live_source, media_id=media_id, session_id=session_id)

    def _handle_submit_action(
        self,
        session_id: str,
        *,
        run_id: str | None,
        fast_ack: bool = False,
    ) -> None:
        self._handle_write_submission(
            session_id=session_id,
            run_id=run_id,
            submitter=self._app.action_submitter,
            not_configured_error="action_submitter_not_configured",
            invalid_payload_error="invalid_action_payload",
            submit_failed_error="action_submit_failed",
            fast_ack=fast_ack,
        )

    def _handle_action_websocket(self, session_id: str, *, run_id: str | None) -> None:
        if self._app.action_submitter is None:
            self._send_json({"error": "action_submitter_not_configured"}, status=HTTPStatus.NOT_IMPLEMENTED)
            return
        if not self._supports_realtime_input_websocket(session_id, run_id=run_id):
            self._send_json({"error": "realtime_input_websocket_unavailable"}, status=HTTPStatus.NOT_FOUND)
            return
        if not _is_websocket_upgrade_request(self.headers):
            self._send_json({"error": "websocket_upgrade_required"}, status=HTTPStatus.BAD_REQUEST)
            return
        websocket_key = str(self.headers.get("Sec-WebSocket-Key") or "").strip()
        if websocket_key == "":
            self._send_json({"error": "missing_websocket_key"}, status=HTTPStatus.BAD_REQUEST)
            return

        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", _build_websocket_accept(websocket_key))
        self.end_headers()
        self.close_connection = True

        try:
            while True:
                frame = _read_websocket_frame(self.rfile)
                if frame is None:
                    break
                opcode, payload = frame
                if opcode == _WS_OPCODE_CLOSE:
                    _write_websocket_frame(self.wfile, opcode=_WS_OPCODE_CLOSE, payload=b"")
                    break
                if opcode == _WS_OPCODE_PING:
                    _write_websocket_frame(self.wfile, opcode=_WS_OPCODE_PONG, payload=payload)
                    continue
                if opcode != _WS_OPCODE_TEXT:
                    continue
                try:
                    message = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(message, dict):
                    continue
                self._app.action_submitter(session_id, run_id, dict(message))
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return
        except Exception:
            logger.exception("Realtime input websocket failed for {}", session_id)

    def _handle_submit_chat(self, session_id: str, *, run_id: str | None) -> None:
        self._handle_write_submission(
            session_id=session_id,
            run_id=run_id,
            submitter=self._app.chat_submitter,
            not_configured_error="chat_submitter_not_configured",
            invalid_payload_error="invalid_chat_payload",
            submit_failed_error="chat_submit_failed",
        )

    def _handle_submit_control(self, session_id: str, *, run_id: str | None) -> None:
        self._handle_write_submission(
            session_id=session_id,
            run_id=run_id,
            submitter=self._app.control_submitter,
            not_configured_error="control_submitter_not_configured",
            invalid_payload_error="invalid_control_payload",
            submit_failed_error="control_submit_failed",
        )

    def _handle_write_submission(
        self,
        *,
        session_id: str,
        run_id: str | None,
        submitter: ActionSubmitter | ChatSubmitter | ControlSubmitter | None,
        not_configured_error: str,
        invalid_payload_error: str,
        submit_failed_error: str,
        fast_ack: bool = False,
    ) -> None:
        try:
            payload = self._read_json_object()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if submitter is None:
            self._send_json({"error": not_configured_error}, status=HTTPStatus.NOT_IMPLEMENTED)
            return

        try:
            receipt = submitter(session_id, run_id, payload)
        except ValueError:
            self._send_json({"error": invalid_payload_error}, status=HTTPStatus.BAD_REQUEST)
            return
        except Exception:
            logger.exception("Failed write submission {} for {}", submit_failed_error, session_id)
            self._send_json({"error": submit_failed_error}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if fast_ack and str(getattr(receipt, "state", "")).lower() in {"accepted", "committed"}:
            self._send_empty(status=HTTPStatus.NO_CONTENT)
            return
        self._send_json(receipt.to_dict(), status=HTTPStatus.OK)

    def _handle_app_request(self, request_path: str) -> None:
        app_dir = self._app.app_dir
        if app_dir is None or not app_dir.exists():
            self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)
            return

        target = _resolve_app_path(app_dir, request_path)
        if target is None or not target.exists() or not target.is_file():
            self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)
            return

        mime_type, _ = guess_type(target.name)
        resolved_mime_type = mime_type or "application/octet-stream"
        if resolved_mime_type.startswith("text/html"):
            resolved_mime_type = "text/html; charset=utf-8"
        elif resolved_mime_type.startswith("text/css"):
            resolved_mime_type = "text/css; charset=utf-8"
        elif resolved_mime_type in {"text/javascript", "application/javascript"}:
            resolved_mime_type = "application/javascript; charset=utf-8"
        self._send_bytes(target.read_bytes(), status=HTTPStatus.OK, mime_type=resolved_mime_type)

    def _read_json_object(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("invalid_json") from exc
        if not isinstance(payload, dict):
            raise ValueError("json_body_must_be_object")
        return dict(payload)

    def _require_manifest_path(self, session_id: str, *, run_id: str | None) -> Path | None:
        try:
            manifest_path = self._app.resolve_manifest_path(session_id, run_id=run_id)
        except SessionResolutionError as exc:
            payload: dict[str, Any] = {"error": exc.error_code}
            if exc.session_id is not None:
                payload["sessionId"] = exc.session_id
            self._send_json(payload, status=exc.status)
            return None
        if manifest_path is None or not manifest_path.exists():
            self._send_json({"error": "session_not_found"}, status=HTTPStatus.NOT_FOUND)
            return None
        return manifest_path

    def _query_run_id(self, parsed_url) -> str | None:
        params = parse_qs(parsed_url.query)
        run_id = _first_param(params, "run_id")
        return None if run_id is None else str(run_id)

    def _query_fast_ack(self, parsed_url) -> bool:
        params = parse_qs(parsed_url.query)
        raw_value = _first_param(params, "fast")
        if raw_value is None:
            return False
        return str(raw_value).strip().lower() not in {"0", "false", "no", "off"}

    def _supports_realtime_input_websocket(self, session_id: str, *, run_id: str | None) -> bool:
        live_source = self._app.resolve_live_session(session_id, run_id=run_id)
        if live_source is None:
            return False
        try:
            session = live_source.load_session()
        except Exception:
            logger.exception("Failed to inspect live session {} for realtime websocket", session_id)
            return False
        capabilities = dict(getattr(session, "capabilities", {}) or {})
        return (
            str(getattr(session, "lifecycle", "")).strip() == "live_running"
            and capabilities.get("supportsRealtimeInputWebSocket") is True
        )

    def _validate_route_tokens(self, session_id: str, *, run_id: str | None) -> tuple[str, str | None] | None:
        try:
            validated_session_id = _validate_lookup_token(session_id, error_code="invalid_session_id")
            validated_run_id = None
            if run_id is not None:
                validated_run_id = _validate_lookup_token(run_id, error_code="invalid_run_id")
        except SessionResolutionError as exc:
            payload: dict[str, Any] = {"error": exc.error_code}
            if exc.session_id is not None:
                payload["sessionId"] = exc.session_id
            self._send_json(payload, status=exc.status)
            return None
        return validated_session_id, validated_run_id

    @property
    def _app(self) -> ArenaVisualHTTPServer:
        return getattr(self.server, "arena_visual_server_ref")  # type: ignore[no-any-return, attr-defined]

    def _send_cors_headers(self) -> None:
        allow_origin = getattr(self.server, "allow_origin", "*")  # type: ignore[attr-defined]
        self.send_header("Access-Control-Allow-Origin", str(allow_origin))
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, payload: Mapping[str, Any], *, status: HTTPStatus) -> None:
        safe_payload = to_scene_json_safe(dict(payload))
        body = json.dumps(safe_payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, payload: bytes, *, status: HTTPStatus, mime_type: str) -> None:
        body = bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, *, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self._send_cors_headers()
        self.end_headers()

    def _send_sse_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        data = json.dumps(to_scene_json_safe(dict(payload)), ensure_ascii=False)
        self.wfile.write(f"event: {event_type}\n".encode("utf-8"))
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream_low_latency_media(
        self,
        *,
        live_source: ArenaVisualLiveSessionSource,
        media_id: str,
        session_id: str,
    ) -> None:
        boundary = b"frame"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self._send_cors_headers()
        self.end_headers()

        last_payload: bytes | None = None
        sent_frames = 0
        idle_rounds_after_end = 0
        while True:
            try:
                frame_payload = live_source.load_stream_frame(media_id)
            except Exception:
                logger.exception("Failed to stream live media {} for {}", media_id, session_id)
                break
            if frame_payload is None:
                break
            content, mime_type = frame_payload
            frame_changed = content != last_payload
            if frame_changed:
                last_payload = content
                sent_frames += 1
                try:
                    self.wfile.write(b"--" + boundary + b"\r\n")
                    part_mime_type = str(mime_type or "application/octet-stream")
                    self.wfile.write(f"Content-Type: {part_mime_type}\r\n".encode("utf-8"))
                    self.wfile.write(f"Content-Length: {len(content)}\r\n\r\n".encode("utf-8"))
                    self.wfile.write(content)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    break

            try:
                header = live_source.load_live_header()
                lifecycle = str(header.get("lifecycle", "")).strip() or None
            except Exception:
                lifecycle = None
            if lifecycle != "live_running":
                if not frame_changed and sent_frames > 0:
                    idle_rounds_after_end += 1
                else:
                    idle_rounds_after_end = 0
                if idle_rounds_after_end >= 2:
                    break
            try:
                time.sleep(_LOW_LATENCY_STREAM_POLL_INTERVAL_S)
            except Exception:
                break

    def _match_session_route(self, path: str) -> tuple[str, tuple[str, ...]] | None:
        parts = tuple(unquote(part) for part in path.split("/") if part)
        if len(parts) < 3:
            return None
        if parts[0] != "arena_visual" or parts[1] != "sessions":
            return None
        return parts[2], parts[3:]


class ArenaVisualHTTPServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8010,
        *,
        base_dir: str | Path = ".",
        query_service: ArenaVisualGatewayQueryService | None = None,
        manifest_resolver: ManifestResolver | None = None,
        action_submitter: ActionSubmitter | None = None,
        chat_submitter: ChatSubmitter | None = None,
        control_submitter: ControlSubmitter | None = None,
        allow_origin: str = "*",
        app_dir: str | Path | None = None,
        live_registry: ArenaVisualLiveRegistry | None = None,
    ) -> None:
        self._host = str(host)
        self._port = int(port)
        self._base_dir = Path(base_dir).expanduser().resolve()
        resolved_app_dir = _default_app_dir() if app_dir is None else Path(app_dir).expanduser().resolve()
        self._app_dir = resolved_app_dir if resolved_app_dir.exists() else None
        self._query_service = query_service or ArenaVisualGatewayQueryService(
            visualization_spec_resolver=resolve_registered_visualization_spec
        )
        self._manifest_resolver = manifest_resolver or build_session_manifest_resolver(self._base_dir)
        self._live_registry = live_registry or ArenaVisualLiveRegistry()
        self._action_submitter = action_submitter
        self._chat_submitter = chat_submitter
        self._control_submitter = control_submitter
        self._allow_origin = str(allow_origin)
        self._server = ThreadingHTTPServer((self._host, self._port), ArenaVisualRequestHandler)
        setattr(self._server, "allow_origin", self._allow_origin)
        setattr(self._server, "arena_visual_server_ref", self)
        self._thread: Thread | None = None

    @property
    def action_submitter(self) -> ActionSubmitter | None:
        return self._action_submitter

    @property
    def chat_submitter(self) -> ChatSubmitter | None:
        return self._chat_submitter

    @property
    def control_submitter(self) -> ControlSubmitter | None:
        return self._control_submitter

    @property
    def query_service(self) -> ArenaVisualGatewayQueryService:
        return self._query_service

    @property
    def server_address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    @property
    def app_dir(self) -> Path | None:
        return self._app_dir

    def register_live_session(self, source: ArenaVisualLiveSessionSource) -> None:
        self._live_registry.register(source)

    def unregister_live_session(self, *, session_id: str, run_id: str | None = None) -> None:
        self._live_registry.unregister(session_id=session_id, run_id=run_id)

    def resolve_live_session(
        self,
        session_id: str,
        *,
        run_id: str | None = None,
    ) -> ArenaVisualLiveSessionSource | None:
        return self._live_registry.resolve(session_id=session_id, run_id=run_id)

    def resolve_manifest_path(self, session_id: str, *, run_id: str | None = None) -> Path | None:
        resolved = self._manifest_resolver(session_id, run_id)
        if resolved is None:
            return None
        return Path(resolved).expanduser().resolve()

    def build_viewer_url(self, session_id: str, *, run_id: str | None = None) -> str:
        host, port = self.server_address
        base_url = f"http://{host}:{port}/sessions/{quote(str(session_id), safe='')}"
        if run_id is None or str(run_id).strip() == "":
            return base_url
        return f"{base_url}?run_id={quote(str(run_id), safe='')}"

    def serve_forever(self) -> None:
        logger.info(
            "Arena visual server listening on http://{}:{} (base={})",
            self._host,
            self._port,
            self._base_dir,
        )
        self._server.serve_forever()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)


def _should_suppress_access_log(*, command: Any, path: Any, message: str) -> bool:
    normalized_command = str(command or "").strip().upper()
    normalized_path = str(path or "").strip()
    if normalized_command not in {"GET", "HEAD"}:
        return False
    if normalized_path == "":
        return False
    status_code = _extract_access_log_status_code(message)
    if status_code is None or status_code >= 500:
        return False
    return (
        normalized_path == "/"
        or normalized_path.startswith("/sessions/")
        or normalized_path.startswith("/assets/")
        or normalized_path.startswith("/arena_visual/sessions/")
    )


def _extract_access_log_status_code(message: str) -> int | None:
    parts = str(message).rsplit(" ", 2)
    if len(parts) < 2:
        return None
    try:
        return int(parts[-2])
    except (TypeError, ValueError):
        return None


def _load_media_content(*, manifest_path: Path, media_url: str | None) -> tuple[bytes, str | None]:
    url = str(media_url or "").strip()
    if not url:
        raise FileNotFoundError("missing_media_url")
    if url.startswith("data:"):
        return _decode_data_url(url)
    if url.lower().startswith(("http://", "https://")):
        raise ValueError("media_content_external_url_unsupported")

    session_dir = manifest_path.parent
    replay_dir = manifest_path.parents[2] if len(manifest_path.parents) >= 3 else session_dir
    for base_dir in (session_dir, replay_dir):
        candidate = (base_dir / url).resolve()
        if not _is_within_root(candidate, base_dir):
            continue
        if candidate.exists() and candidate.is_file():
            mime_type, _ = guess_type(candidate.name)
            return candidate.read_bytes(), mime_type
    raise FileNotFoundError(url)


def _decode_data_url(url: str) -> tuple[bytes, str | None]:
    header, separator, data = url.partition(",")
    if separator == "":
        raise ValueError("invalid_media_data_url")
    mime_type = header[5:].split(";", 1)[0].strip() or None
    if ";base64" in header:
        try:
            return base64.b64decode(data), mime_type
        except (ValueError, binascii.Error) as exc:
            raise ValueError("invalid_media_data_url") from exc
    return unquote_to_bytes(data), mime_type


def _is_within_root(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _resolve_app_path(app_dir: Path, request_path: str) -> Path | None:
    normalized_path = str(request_path or "/").split("?", 1)[0]
    stripped = unquote(normalized_path).lstrip("/")
    if stripped == "" or stripped.endswith("/"):
        return app_dir / "index.html"

    candidate = (app_dir / stripped).resolve()
    if _is_within_root(candidate, app_dir) and candidate.exists() and candidate.is_file():
        return candidate

    if "." not in Path(stripped).name:
        return app_dir / "index.html"
    return None


def _is_websocket_upgrade_request(headers) -> bool:
    upgrade = str(headers.get("Upgrade") or "").strip().lower()
    connection = str(headers.get("Connection") or "").strip().lower()
    return upgrade == "websocket" and "upgrade" in connection


def _build_websocket_accept(key: str) -> str:
    digest = hashlib.sha1(f"{key}{_WEBSOCKET_ACCEPT_GUID}".encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


def _read_exact(reader, size: int) -> bytes | None:
    if size <= 0:
        return b""
    try:
        payload = reader.read(size)
    except Exception:
        return None
    if payload is None or len(payload) != size:
        return None
    return payload


def _read_websocket_frame(reader) -> tuple[int, bytes] | None:
    header = _read_exact(reader, 2)
    if header is None:
        return None
    first_byte, second_byte = header
    opcode = first_byte & 0x0F
    masked = (second_byte & 0x80) != 0
    payload_length = second_byte & 0x7F
    if payload_length == 126:
        extended = _read_exact(reader, 2)
        if extended is None:
            return None
        payload_length = int.from_bytes(extended, "big")
    elif payload_length == 127:
        extended = _read_exact(reader, 8)
        if extended is None:
            return None
        payload_length = int.from_bytes(extended, "big")
    mask_key = _read_exact(reader, 4) if masked else b""
    if masked and mask_key is None:
        return None
    payload = _read_exact(reader, payload_length)
    if payload is None:
        return None
    if masked:
        payload = bytes(
            value ^ mask_key[index % 4]
            for index, value in enumerate(payload)
        )
    return opcode, payload


def _write_websocket_frame(writer, *, opcode: int, payload: bytes) -> None:
    header = bytearray([0x80 | (opcode & 0x0F)])
    payload_length = len(payload)
    if payload_length < 126:
        header.append(payload_length)
    elif payload_length < (1 << 16):
        header.append(126)
        header.extend(payload_length.to_bytes(2, "big"))
    else:
        header.append(127)
        header.extend(payload_length.to_bytes(8, "big"))
    writer.write(bytes(header))
    writer.write(payload)
    writer.flush()


def _default_app_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[5]
    return (repo_root / "frontend" / "arena-visual" / "dist").resolve()


__all__ = [
    "ActionSubmitter",
    "ChatSubmitter",
    "ControlSubmitter",
    "ArenaVisualHTTPServer",
    "ArenaVisualRequestHandler",
    "ManifestResolver",
    "build_session_manifest_resolver",
]
