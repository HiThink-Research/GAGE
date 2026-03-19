from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlparse

from loguru import logger

_LOOPBACK_ORIGIN_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _sanitize_sample_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return cleaned.strip("_") or "unknown"


def _first_param(params: dict[str, list[str]], key: str) -> Optional[str]:
    values = params.get(key)
    if not values:
        return None
    return values[0]


def _parse_bool(value: Optional[str], *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalize_allowed_origins(origins: Optional[Sequence[str]]) -> tuple[str, ...]:
    """Normalize configured replay CORS origins."""

    if not origins:
        return ()
    normalized: list[str] = []
    for value in origins:
        for item in str(value).split(","):
            candidate = item.strip()
            if candidate:
                normalized.append(candidate)
    return tuple(dict.fromkeys(normalized))


def _is_origin_allowed(origin: Optional[str], allowed_origins: Optional[Sequence[str]]) -> bool:
    """Return whether the request origin is allowed to access replay endpoints."""

    if not origin:
        return True
    normalized_origin = str(origin).strip()
    if not normalized_origin:
        return True

    parsed_origin = urlparse(normalized_origin)
    hostname = str(parsed_origin.hostname or "").strip().lower()
    if hostname in _LOOPBACK_ORIGIN_HOSTS:
        return True

    normalized_allowed = _normalize_allowed_origins(allowed_origins)
    if "*" in normalized_allowed:
        return True

    candidate_origin = (
        f"{parsed_origin.scheme}://{parsed_origin.netloc}"
        if parsed_origin.scheme and parsed_origin.netloc
        else normalized_origin
    )
    return normalized_origin in normalized_allowed or candidate_origin in normalized_allowed


def _safe_resolve(base_dir: Path, candidate: Path) -> Optional[Path]:
    base_resolved = base_dir.expanduser().resolve()
    if not candidate.is_absolute():
        candidate = base_resolved / candidate
    try:
        resolved = candidate.resolve()
    except FileNotFoundError:
        resolved = candidate.resolve(strict=False)
    if base_resolved not in resolved.parents and resolved != base_resolved:
        return None
    return resolved


def _resolve_replay_path(
    base_dir: Path,
    *,
    replay_path: Optional[str],
    run_id: Optional[str],
    sample_id: Optional[str],
) -> Optional[Path]:
    base_resolved = base_dir.expanduser().resolve()
    if replay_path:
        return _safe_resolve(base_resolved, Path(replay_path))

    if not run_id:
        return None
    safe_sample_id = _sanitize_sample_id(sample_id or "unknown")
    replay_root = base_resolved / run_id / "replays"
    v1_manifest = replay_root / safe_sample_id / "replay.json"
    if v1_manifest.exists():
        return v1_manifest
    legacy = replay_root / f"doudizhu_replay_{safe_sample_id}.json"
    if legacy.exists():
        return legacy
    # NOTE: Return preferred v1 location even when the file does not exist.
    return v1_manifest


def _resolve_events_path(base_dir: Path, *, replay_file: Path, payload: dict[str, Any]) -> Optional[Path]:
    if payload.get("schema") != "gage_replay/v1":
        return None
    recording = payload.get("recording")
    files = payload.get("files")
    events_rel = None
    if isinstance(recording, dict):
        events_rel = recording.get("events_path")
    if not events_rel and isinstance(files, dict):
        events_rel = files.get("events")
    events_rel = str(events_rel or "events.jsonl")
    return _safe_resolve(base_dir, replay_file.parent / events_rel)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("replay_payload_must_be_object")
    return payload


def _iter_events_jsonl(
    path: Path,
    *,
    event_type: Optional[str] = None,
) -> Iterator[dict[str, Any]]:
    """Yield replay events from a JSONL stream one line at a time."""

    normalized_event_type = str(event_type).strip().lower() if event_type else None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                continue
            if normalized_event_type is not None:
                current_type = str(parsed.get("type") or "").strip().lower()
                if current_type != normalized_event_type:
                    continue
            yield parsed


def _load_events_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_events_jsonl(path))


def _load_events_page_jsonl(
    path: Path,
    *,
    offset: int = 0,
    limit: Optional[int] = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Load one page of replay events from JSONL without materializing the whole file."""

    if offset < 0:
        raise ValueError("events_offset_must_be_non_negative")
    if limit is not None and limit < 0:
        raise ValueError("events_limit_must_be_non_negative")

    events: list[dict[str, Any]] = []
    matched_count = 0
    has_more = False

    for event in _iter_events_jsonl(path):
        if matched_count < offset:
            matched_count += 1
            continue
        if limit is not None and len(events) >= limit:
            has_more = True
            break
        events.append(event)
        matched_count += 1

    return events, has_more


def _resolve_frame_events(
    base_dir: Path,
    *,
    replay_file: Path,
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], Optional[str]]:
    if payload.get("schema") != "gage_replay/v1":
        return [], "frame_events_unsupported_schema"
    events_path = _resolve_events_path(base_dir, replay_file=replay_file, payload=payload)
    if events_path is None or not events_path.exists():
        return [], "frame_events_not_found"
    try:
        frame_events = [dict(event) for event in _iter_events_jsonl(events_path, event_type="frame")]
    except Exception:
        return [], "frame_events_read_failed"
    if not frame_events:
        return [], "frame_events_not_found"
    return frame_events, None


def _select_frame_event(
    frame_events: Sequence[Mapping[str, Any]],
    *,
    seq: Optional[int],
    index: Optional[int],
) -> Optional[dict[str, Any]]:
    if seq is not None:
        for event in frame_events:
            event_seq = _parse_int(str(event.get("seq")) if event.get("seq") is not None else None)
            if event_seq == seq:
                return dict(event)
        return None
    normalized_index = 0 if index is None else int(index)
    if normalized_index < 0 or normalized_index >= len(frame_events):
        return None
    return dict(frame_events[normalized_index])


def _resolve_frame_image_path(
    base_dir: Path,
    *,
    replay_file: Path,
    frame_event: Mapping[str, Any],
) -> Optional[Path]:
    image = frame_event.get("image")
    if not isinstance(image, Mapping):
        return None
    image_path = image.get("path")
    if not image_path:
        return None
    resolved = _safe_resolve(base_dir, replay_file.parent / Path(str(image_path)))
    if resolved is None:
        return None
    replay_dir = replay_file.parent.resolve()
    if replay_dir not in resolved.parents and resolved != replay_dir:
        return None
    return resolved


def _content_type_from_path(path: Path) -> str:
    suffix = str(path.suffix).lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "application/octet-stream"


def _legacy_payload_to_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    moves = payload.get("moves")
    if isinstance(moves, list):
        for seq, move in enumerate(moves, start=1):
            if isinstance(move, dict):
                events.append(
                    {
                        "type": "action",
                        "seq": seq,
                        "actor": move.get("player_id") or move.get("player") or move.get("playerIdx") or "unknown",
                        "move": move.get("action_text") or move.get("move") or move.get("action") or "",
                        "raw": move.get("raw") or move.get("action_text") or move.get("move") or "",
                        "ts_ms": move.get("timestamp_ms") or move.get("ts_ms"),
                        "meta": dict(move),
                    }
                )
    move_history = payload.get("moveHistory") or payload.get("move_history")
    if isinstance(move_history, list):
        for seq, move in enumerate(move_history, start=len(events) + 1):
            if isinstance(move, dict):
                events.append(
                    {
                        "type": "action",
                        "seq": seq,
                        "actor": move.get("playerIdx") or move.get("player_idx") or "unknown",
                        "move": move.get("move") or move.get("action_text") or "",
                        "raw": move.get("raw") or move.get("move") or "",
                        "ts_ms": move.get("timestamp_ms") or move.get("timestampMs"),
                        "meta": dict(move),
                    }
                )
    winner = payload.get("winner")
    result = payload.get("result")
    reason = payload.get("reason") or payload.get("result_reason") or payload.get("end_reason")
    if winner is not None or result is not None or reason is not None:
        events.append(
            {
                "type": "result",
                "seq": len(events) + 1,
                "winner": winner,
                "result": result,
                "reason": reason,
            }
        )
    return events


class ReplayRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves unified and legacy replay artifacts."""

    server_version = "GAGEReplayServer/1.0"

    def do_OPTIONS(self) -> None:
        if self._reject_disallowed_origin():
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - matching BaseHTTPRequestHandler
        if self._reject_disallowed_origin():
            return
        parsed = urlparse(self.path)
        if parsed.path == "/tournament/replay":
            self._handle_replay(parsed)
            return
        if parsed.path == "/tournament/replay/events":
            self._handle_events(parsed)
            return
        if parsed.path == "/tournament/replay/frame":
            self._handle_frame(parsed)
            return
        self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)

    def _handle_replay(self, parsed_url) -> None:
        params = parse_qs(parsed_url.query)
        replay_path = _first_param(params, "replay_path")
        run_id = _first_param(params, "run_id")
        sample_id = _first_param(params, "sample_id")
        include_events = _parse_bool(_first_param(params, "include_events"))
        base_dir = Path(self.server.base_dir)  # type: ignore[attr-defined]
        target = _resolve_replay_path(
            base_dir,
            replay_path=replay_path,
            run_id=run_id,
            sample_id=sample_id,
        )
        if target is None:
            self._send_json({"error": "Missing replay parameters"}, status=HTTPStatus.BAD_REQUEST)
            return
        if not target.exists():
            self._send_json({"error": f"Replay not found: {target}"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            payload = _load_json(target)
        except Exception as exc:
            self._send_json({"error": f"Failed to read replay: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if include_events:
            events_path = _resolve_events_path(base_dir, replay_file=target, payload=payload)
            if events_path is not None and events_path.exists():
                try:
                    payload["events"] = _load_events_jsonl(events_path)
                except Exception as exc:
                    self._send_json(
                        {"error": f"Failed to read replay events: {exc}"},
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                    return
        self._send_json(payload, status=HTTPStatus.OK)

    def _handle_events(self, parsed_url) -> None:
        params = parse_qs(parsed_url.query)
        replay_path = _first_param(params, "replay_path")
        run_id = _first_param(params, "run_id")
        sample_id = _first_param(params, "sample_id")
        offset = _parse_int(_first_param(params, "offset"))
        limit = _parse_int(_first_param(params, "limit"))
        if offset is not None and offset < 0:
            self._send_json({"error": "Invalid offset"}, status=HTTPStatus.BAD_REQUEST)
            return
        if limit is not None and limit < 0:
            self._send_json({"error": "Invalid limit"}, status=HTTPStatus.BAD_REQUEST)
            return
        base_dir = Path(self.server.base_dir)  # type: ignore[attr-defined]
        target = _resolve_replay_path(
            base_dir,
            replay_path=replay_path,
            run_id=run_id,
            sample_id=sample_id,
        )
        if target is None:
            self._send_json({"error": "Missing replay parameters"}, status=HTTPStatus.BAD_REQUEST)
            return
        if not target.exists():
            self._send_json({"error": f"Replay not found: {target}"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            payload = _load_json(target)
        except Exception as exc:
            self._send_json({"error": f"Failed to read replay: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        events: list[dict[str, Any]]
        has_more = False
        normalized_offset = 0 if offset is None else offset
        if payload.get("schema") == "gage_replay/v1":
            events_path = _resolve_events_path(base_dir, replay_file=target, payload=payload)
            if events_path is None or not events_path.exists():
                self._send_json({"error": "Replay events not found"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                events, has_more = _load_events_page_jsonl(
                    events_path,
                    offset=normalized_offset,
                    limit=limit,
                )
            except Exception as exc:
                self._send_json(
                    {"error": f"Failed to read replay events: {exc}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
        else:
            events = _legacy_payload_to_events(payload)
            if normalized_offset:
                events = events[normalized_offset:]
            if limit is not None:
                has_more = len(events) > limit
                events = events[:limit]

        response = {
            "schema": "gage_replay/events.v1",
            "replay_path": str(target),
            "events": events,
        }
        if offset is not None or limit is not None:
            response["offset"] = normalized_offset
            response["limit"] = limit
            response["has_more"] = has_more
        self._send_json(response, status=HTTPStatus.OK)

    def _handle_frame(self, parsed_url) -> None:
        params = parse_qs(parsed_url.query)
        replay_path = _first_param(params, "replay_path")
        run_id = _first_param(params, "run_id")
        sample_id = _first_param(params, "sample_id")
        seq = _parse_int(_first_param(params, "seq"))
        index = _parse_int(_first_param(params, "index"))
        frame_format = str(_first_param(params, "format") or "image").strip().lower()
        include_event = _parse_bool(_first_param(params, "include_event"), default=False)
        if seq is not None and seq < 0:
            self._send_json({"error": "Invalid seq"}, status=HTTPStatus.BAD_REQUEST)
            return
        if index is not None and index < 0:
            self._send_json({"error": "Invalid index"}, status=HTTPStatus.BAD_REQUEST)
            return

        base_dir = Path(self.server.base_dir)  # type: ignore[attr-defined]
        replay_file = _resolve_replay_path(
            base_dir,
            replay_path=replay_path,
            run_id=run_id,
            sample_id=sample_id,
        )
        if replay_file is None:
            self._send_json({"error": "Missing replay parameters"}, status=HTTPStatus.BAD_REQUEST)
            return
        if not replay_file.exists():
            self._send_json({"error": f"Replay not found: {replay_file}"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            payload = _load_json(replay_file)
        except Exception as exc:
            self._send_json({"error": f"Failed to read replay: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        frame_events, frame_error = _resolve_frame_events(
            base_dir,
            replay_file=replay_file,
            payload=payload,
        )
        if frame_error == "frame_events_unsupported_schema":
            self._send_json({"error": "Replay does not support frame events"}, status=HTTPStatus.BAD_REQUEST)
            return
        if frame_error == "frame_events_not_found":
            self._send_json({"error": "Replay frame events not found"}, status=HTTPStatus.NOT_FOUND)
            return
        if frame_error is not None:
            self._send_json({"error": "Failed to read replay frame events"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        frame_event = _select_frame_event(frame_events, seq=seq, index=index)
        if frame_event is None:
            self._send_json({"error": "Replay frame not found"}, status=HTTPStatus.NOT_FOUND)
            return

        if frame_format == "json":
            response: dict[str, Any] = {
                "schema": "gage_replay/frame.v1",
                "replay_path": str(replay_file),
                "frame": frame_event,
            }
            if include_event:
                response["events_count"] = len(frame_events)
            self._send_json(response, status=HTTPStatus.OK)
            return

        frame_path = _resolve_frame_image_path(
            base_dir,
            replay_file=replay_file,
            frame_event=frame_event,
        )
        if frame_path is None or not frame_path.exists():
            self._send_json({"error": "Replay frame image not found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            frame_bytes = frame_path.read_bytes()
        except Exception as exc:
            self._send_json(
                {"error": f"Failed to read replay frame image: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_binary(
            frame_bytes,
            status=HTTPStatus.OK,
            content_type=_content_type_from_path(frame_path),
        )

    def log_message(self, format: str, *args) -> None:  # noqa: A003 - match base signature
        logger.debug("ReplayServer {}", format % args)

    def _reject_disallowed_origin(self) -> bool:
        """Reject cross-origin requests that are outside the replay allowlist."""

        request_origin = self.headers.get("Origin")
        allowed_origins = getattr(self.server, "allowed_origins", ())  # type: ignore[attr-defined]
        if _is_origin_allowed(request_origin, allowed_origins):
            return False
        self._send_json({"error": "Origin not allowed"}, status=HTTPStatus.FORBIDDEN)
        return True

    def _send_cors_headers(self) -> None:
        """Emit CORS headers for allowed replay requests."""

        request_origin = self.headers.get("Origin")
        allowed_origins = getattr(self.server, "allowed_origins", ())  # type: ignore[attr-defined]
        if request_origin and _is_origin_allowed(request_origin, allowed_origins):
            self.send_header("Access-Control-Allow-Origin", str(request_origin))
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, payload: dict[str, object], *, status: HTTPStatus) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(self, body: bytes, *, status: HTTPStatus, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", str(content_type))
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the replay server."""

    parser = argparse.ArgumentParser(description="Serve replay manifest and events for arena games.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument(
        "--replay-dir",
        default=os.environ.get("GAGE_EVAL_SAVE_DIR", "./runs"),
        help="Base directory that contains run outputs.",
    )
    parser.add_argument(
        "--allowed-origin",
        action="append",
        dest="allowed_origins",
        default=[],
        help="Explicitly allow one non-loopback replay origin. Repeat to add multiple values.",
    )
    return parser


def main() -> None:
    """Run the replay server."""

    parser = build_arg_parser()
    args = parser.parse_args()
    base_dir = Path(args.replay_dir).expanduser()
    server = HTTPServer((args.host, args.port), ReplayRequestHandler)
    setattr(server, "base_dir", str(base_dir))
    setattr(server, "allowed_origins", _normalize_allowed_origins(args.allowed_origins))
    logger.info(
        "Replay server listening on http://{}:{} (base={} allowed_origins={})",
        args.host,
        args.port,
        base_dir,
        list(getattr(server, "allowed_origins", ())),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Replay server stopped")


if __name__ == "__main__":
    main()
