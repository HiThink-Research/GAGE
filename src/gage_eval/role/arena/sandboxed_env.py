"""Sandbox-backed arena environment implementation."""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from gage_eval.role.arena.types import ArenaAction, ArenaObservation, GameResult
from gage_eval.sandbox.base import BaseSandbox


class SandboxedArenaEnvironment:
    """Proxy arena environment operations through sandbox.exec().

    The sandbox is expected to host a long-lived TCP daemon that listens on a
    local port and preserves game state across multiple exec calls.
    """

    DAEMON_HOST = "localhost"
    DAEMON_PORT = 9999

    def __init__(
        self,
        sandbox: BaseSandbox,
        env_kwargs: Dict[str, Any],
        *,
        timeout_s: int = 30,
    ) -> None:
        self._sandbox = sandbox
        self._env_kwargs = dict(env_kwargs or {})
        self._timeout_s = max(1, int(timeout_s))
        self._initialized = False

    def reset(self) -> None:
        """Reset the remote arena environment."""

        if not self._initialized:
            self._rpc("init", self._env_kwargs)
            self._initialized = True
        self._rpc("reset", {})

    def get_active_player(self) -> str:
        """Return the active player id from the daemon state."""

        response = self._rpc("get_active_player", {})
        return str(response.get("player_id") or "")

    def observe(self, player: str) -> ArenaObservation:
        """Return an observation for the requested player id."""

        response = self._rpc("observe", {"player": player})
        return _deserialize_observation(response.get("observation") or {})

    def apply(self, action: ArenaAction) -> Optional[GameResult]:
        """Apply an action and optionally return a terminal result."""

        response = self._rpc("apply", {"action": _serialize_payload(action)})
        game_result = response.get("game_result")
        if game_result is None:
            return None
        return _deserialize_game_result(game_result)

    def is_terminal(self) -> bool:
        """Return whether the daemon reports a terminal game state."""

        response = self._rpc("is_terminal", {})
        return bool(response.get("terminal", False))

    def build_result(self, *, result: str, reason: Optional[str]) -> GameResult:
        """Build a terminal result through the daemon interface."""

        response = self._rpc("build_result", {"result": result, "reason": reason})
        return _deserialize_game_result(response.get("game_result") or {})

    def _rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = json.dumps(
            {"method": method, "params": _serialize_payload(params)},
            ensure_ascii=True,
            separators=(",", ":"),
        )
        command = _build_rpc_command(
            payload, host=self.DAEMON_HOST, port=self.DAEMON_PORT
        )
        result = self._sandbox.exec(command, timeout=self._timeout_s)
        if result.exit_code != 0:
            raise RuntimeError(f"game_rpc_failed: {result.stderr or result.stdout}")
        stdout = (result.stdout or "").strip()
        if not stdout:
            raise RuntimeError("game_rpc_failed: empty_response")
        response = json.loads(stdout)
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(f"game_daemon_error: {response['error']}")
        if not isinstance(response, dict):
            raise RuntimeError("game_rpc_failed: invalid_response")
        return response


def _build_rpc_command(payload: str, *, host: str, port: int) -> str:
    payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    return (
        "python - <<'PY'\n"
        "import base64\n"
        "import socket\n"
        f"payload = base64.b64decode({payload_b64!r}) + b'\\n'\n"
        f"sock = socket.create_connection(({host!r}, {int(port)}), timeout=5)\n"
        "with sock:\n"
        "    sock.sendall(payload)\n"
        "    chunks = []\n"
        "    while True:\n"
        "        chunk = sock.recv(65536)\n"
        "        if not chunk:\n"
        "            break\n"
        "        chunks.append(chunk)\n"
        "        if b'\\n' in chunk:\n"
        "            break\n"
        "response = b''.join(chunks).decode('utf-8')\n"
        "print(response, end='')\n"
        "PY\n"
    )


def _serialize_payload(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize_payload(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_payload(item) for item in value]
    return value


def _deserialize_observation(payload: Any) -> ArenaObservation:
    data = dict(payload or {}) if isinstance(payload, dict) else {}
    return ArenaObservation(
        board_text=str(data.get("board_text") or data.get("view_text") or ""),
        legal_moves=list(data.get("legal_moves") or []),
        active_player=str(data.get("active_player") or ""),
        last_move=data.get("last_move"),
        metadata=dict(data.get("metadata") or {}),
        view=dict(data.get("view") or {}) or None,
        legal_actions=dict(data.get("legal_actions") or {}) or None,
        context=dict(data.get("context") or {}) or None,
    )


def _deserialize_game_result(payload: Any) -> GameResult:
    data = dict(payload or {}) if isinstance(payload, dict) else {}
    return GameResult(
        winner=data.get("winner"),
        result=str(data.get("result") or ""),
        reason=data.get("reason"),
        move_count=int(data.get("move_count") or 0),
        illegal_move_count=int(data.get("illegal_move_count") or 0),
        final_board=str(data.get("final_board") or ""),
        move_log=list(data.get("move_log") or []),
        rule_profile=data.get("rule_profile"),
        win_direction=data.get("win_direction"),
        line_length=data.get("line_length"),
        replay_path=data.get("replay_path"),
        arena_trace=tuple(data.get("arena_trace") or ()),
    )


__all__ = ["SandboxedArenaEnvironment"]
