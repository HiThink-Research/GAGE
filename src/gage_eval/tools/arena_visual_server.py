from __future__ import annotations

import argparse
import os
from pathlib import Path

from loguru import logger

from gage_eval.role.arena.visualization.http_server import ArenaVisualHTTPServer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve arena visual session artifacts over HTTP."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8010, help="Port to listen on.")
    parser.add_argument(
        "--arena-visual-dir",
        default=os.environ.get("GAGE_EVAL_SAVE_DIR", "./runs"),
        help="Base directory that contains run outputs and arena visual session artifacts.",
    )
    parser.add_argument(
        "--allow-origin",
        default="*",
        help="CORS allow-origin header value.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    server = ArenaVisualHTTPServer(
        host=args.host,
        port=args.port,
        base_dir=Path(args.arena_visual_dir).expanduser(),
        allow_origin=args.allow_origin,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Arena visual server stopped")


if __name__ == "__main__":
    main()
