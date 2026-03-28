from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from mimetypes import guess_type
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, unquote, unquote_to_bytes, urlparse

from loguru import logger

from gage_eval.role.arena.visualization.contracts import ActionIntentReceipt, ObserverRef
from gage_eval.role.arena.visualization.gateway_service import ArenaVisualGatewayQueryService

ManifestResolver = Callable[[str, str | None], str | Path | None]
ActionSubmitter = Callable[[str, str | None, dict[str, Any]], ActionIntentReceipt]
_SESSION_MANIFEST_SUFFIX = Path("arena_visual_session") / "v1" / "manifest.json"


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
        if suffix == ("scene",):
            self._handle_scene(session_id, parsed, run_id=run_id)
            return
        if suffix == ("markers",):
            self._handle_markers(session_id, parsed, run_id=run_id)
            return
        if len(suffix) == 2 and suffix[0] == "media":
            self._handle_media(session_id, parsed, media_id=suffix[1], run_id=run_id)
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
            self._handle_submit_action(session_id, run_id=run_id)
            return
        self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - match base signature
        logger.debug("ArenaVisualServer {}", format % args)

    def _handle_session(self, session_id: str, parsed_url, *, run_id: str | None) -> None:
        manifest_path = self._require_manifest_path(session_id, run_id=run_id)
        if manifest_path is None:
            return
        params = parse_qs(parsed_url.query)
        try:
            observer = _parse_observer_override(params)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            session = self._app.query_service.load_session(manifest_path, observer=observer)
        except Exception:
            logger.exception("Failed to load visual session {}", session_id)
            self._send_json({"error": "session_load_failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(session.to_dict(), status=HTTPStatus.OK)

    def _handle_timeline(self, session_id: str, parsed_url, *, run_id: str | None) -> None:
        manifest_path = self._require_manifest_path(session_id, run_id=run_id)
        if manifest_path is None:
            return
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

    def _handle_scene(self, session_id: str, parsed_url, *, run_id: str | None) -> None:
        manifest_path = self._require_manifest_path(session_id, run_id=run_id)
        if manifest_path is None:
            return
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
        manifest_path = self._require_manifest_path(session_id, run_id=run_id)
        if manifest_path is None:
            return
        params = parse_qs(parsed_url.query)
        marker = str(_first_param(params, "marker") or "").strip()
        if not marker:
            self._send_json({"error": "missing_marker"}, status=HTTPStatus.BAD_REQUEST)
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
        manifest_path = self._require_manifest_path(session_id, run_id=run_id)
        if manifest_path is None:
            return
        params = parse_qs(parsed_url.query)
        serve_content = _first_param(params, "content") in {"1", "true", "yes"}
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

    def _handle_submit_action(self, session_id: str, *, run_id: str | None) -> None:
        try:
            payload = self._read_json_object()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self._app.action_submitter is None:
            self._send_json({"error": "action_submitter_not_configured"}, status=HTTPStatus.NOT_IMPLEMENTED)
            return

        try:
            receipt = self._app.action_submitter(session_id, run_id, payload)
        except ValueError:
            self._send_json({"error": "invalid_action_payload"}, status=HTTPStatus.BAD_REQUEST)
            return
        except Exception:
            logger.exception("Failed to submit action for {}", session_id)
            self._send_json({"error": "action_submit_failed"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(receipt.to_dict(), status=HTTPStatus.OK)

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
        body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
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
        allow_origin: str = "*",
    ) -> None:
        self._host = str(host)
        self._port = int(port)
        self._base_dir = Path(base_dir).expanduser().resolve()
        self._query_service = query_service or ArenaVisualGatewayQueryService()
        self._manifest_resolver = manifest_resolver or build_session_manifest_resolver(self._base_dir)
        self._action_submitter = action_submitter
        self._allow_origin = str(allow_origin)
        self._server = HTTPServer((self._host, self._port), ArenaVisualRequestHandler)
        setattr(self._server, "allow_origin", self._allow_origin)
        setattr(self._server, "arena_visual_server_ref", self)
        self._thread: Thread | None = None

    @property
    def action_submitter(self) -> ActionSubmitter | None:
        return self._action_submitter

    @property
    def query_service(self) -> ArenaVisualGatewayQueryService:
        return self._query_service

    @property
    def server_address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def resolve_manifest_path(self, session_id: str, *, run_id: str | None = None) -> Path | None:
        resolved = self._manifest_resolver(session_id, run_id)
        if resolved is None:
            return None
        return Path(resolved).expanduser().resolve()

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


__all__ = [
    "ActionSubmitter",
    "ArenaVisualHTTPServer",
    "ArenaVisualRequestHandler",
    "ManifestResolver",
    "build_session_manifest_resolver",
]
