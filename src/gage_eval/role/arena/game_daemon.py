"""TCP daemon for sandboxed arena environments.

This daemon is designed to run **inside** a sandbox container, accepting
one connection at a time on a serial accept loop.  It is NOT safe for
concurrent access — callers must serialize requests externally.

Security: when running inside a container, pass ``--host 0.0.0.0`` so
the sandbox exec bridge can reach the daemon.  The port must NOT be
published to the host network; container isolation is the security
boundary.  The default bind address is 127.0.0.1 (loopback only) so
that local-dev usage is safe by default.
"""

from __future__ import annotations

import argparse
import json
import socket
import traceback
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from gage_eval.registry import registry
from gage_eval.role.arena.registry_loader import import_arena_asset_module
from gage_eval.role.arena.types import ArenaAction


class _DaemonState:
    """Per-daemon mutable state.  Avoids module-level globals."""

    __slots__ = ("env",)

    def __init__(self) -> None:
        self.env: Optional[Any] = None


_STATE = _DaemonState()


def handle_request(
    request: Dict[str, Any],
    state: Optional[_DaemonState] = None,
) -> Dict[str, Any]:
    """Handle a single line-delimited JSON-RPC style request."""

    if state is None:
        state = _STATE

    method = str(request.get("method") or "")
    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    if method == "init":
        impl = params.get("impl")
        if not impl:
            raise ValueError("init requires impl")
        try:
            env_cls = registry.get("arena_impls", impl)
        except KeyError:
            import_arena_asset_module("arena_impls", impl)
            env_cls = registry.get("arena_impls", impl)
        init_kwargs = dict(params)
        init_kwargs.pop("impl", None)
        state.env = env_cls(**init_kwargs)
        return {"status": "ok"}
    if state.env is None:
        raise RuntimeError("environment_not_initialized")
    if method == "reset":
        state.env.reset()
        return {"status": "ok"}
    if method == "get_active_player":
        return {"player_id": state.env.get_active_player()}
    if method == "observe":
        return {
            "observation": _serialize(
                state.env.observe(str(params.get("player") or ""))
            )
        }
    if method == "apply":
        action_payload = (
            params.get("action") if isinstance(params.get("action"), dict) else {}
        )
        result = state.env.apply(_deserialize_action(action_payload))
        return {"game_result": _serialize(result) if result is not None else None}
    if method == "is_terminal":
        return {"terminal": bool(state.env.is_terminal())}
    if method == "build_result":
        result = state.env.build_result(
            result=str(params.get("result") or ""),
            reason=params.get("reason"),
        )
        return {"game_result": _serialize(result)}
    raise ValueError(f"unknown method: {method}")


def serve(*, host: str = "127.0.0.1", port: int = 9999) -> None:
    """Start the arena daemon loop (single-connection, serial)."""

    state = _DaemonState()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, int(port)))
    server.listen(1)
    print(f"game_daemon listening on {host}:{port}", flush=True)
    while True:
        conn, _ = server.accept()
        with conn:
            data = b""
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break
            try:
                request = json.loads(data.strip() or b"{}")
                response = handle_request(request, state)
            except Exception as exc:  # pragma: no cover - defensive daemon path
                response = {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            conn.sendall(
                json.dumps(response, ensure_ascii=True).encode("utf-8") + b"\n"
            )


def _deserialize_action(payload: Dict[str, Any]) -> ArenaAction:
    return ArenaAction(
        player=str(payload.get("player") or payload.get("player_id") or ""),
        move=str(payload.get("move") or ""),
        raw=str(payload.get("raw") or payload.get("move") or ""),
        metadata=dict(payload.get("metadata") or {}),
    )


def _serialize(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the arena TCP daemon.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    return parser.parse_args()


def main() -> None:
    """Run the CLI entrypoint."""

    args = _parse_args()
    serve(host=str(args.host), port=int(args.port))


if __name__ == "__main__":
    main()
