from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from loguru import logger


def _sanitize_sample_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return cleaned.strip("_") or "unknown"


def _first_param(params: dict[str, list[str]], key: str) -> Optional[str]:
    values = params.get(key)
    if not values:
        return None
    return values[0]


def _resolve_replay_path(
    base_dir: Path,
    *,
    replay_path: Optional[str],
    run_id: Optional[str],
    sample_id: Optional[str],
) -> Optional[Path]:
    if replay_path:
        candidate = Path(replay_path)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate
        base_resolved = base_dir.resolve()
        if base_resolved not in resolved.parents and resolved != base_resolved:
            return None
        return resolved

    if not run_id:
        return None
    safe_sample_id = _sanitize_sample_id(sample_id or "unknown")
    return base_dir / run_id / "replays" / f"doudizhu_replay_{safe_sample_id}.json"


class ReplayRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves showdown replay JSON files."""

    server_version = "GAGEReplayServer/1.0"

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - matching BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        if parsed.path != "/tournament/replay":
            self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)
            return

        params = parse_qs(parsed.query)
        replay_path = _first_param(params, "replay_path")
        run_id = _first_param(params, "run_id")
        sample_id = _first_param(params, "sample_id")
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
            payload = json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:
            self._send_json({"error": f"Failed to read replay: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(payload, status=HTTPStatus.OK)

    def log_message(self, format: str, *args) -> None:  # noqa: A003 - match base signature
        logger.debug("ReplayServer {}", format % args)

    def _send_json(self, payload: dict[str, object], *, status: HTTPStatus) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the replay server."""

    parser = argparse.ArgumentParser(description="Serve Doudizhu showdown replay JSON files.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument(
        "--replay-dir",
        default=os.environ.get("GAGE_EVAL_SAVE_DIR", "./runs"),
        help="Base directory that contains run outputs.",
    )
    return parser


def main() -> None:
    """Run the replay server."""

    parser = build_arg_parser()
    args = parser.parse_args()
    base_dir = Path(args.replay_dir).expanduser()
    server = HTTPServer((args.host, args.port), ReplayRequestHandler)
    setattr(server, "base_dir", str(base_dir))
    logger.info("Replay server listening on http://{}:{} (base={})", args.host, args.port, base_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Replay server stopped")


if __name__ == "__main__":
    main()
