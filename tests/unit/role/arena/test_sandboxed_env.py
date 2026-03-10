from __future__ import annotations

import base64
import json
import re

import pytest

from gage_eval.role.arena.interfaces import ArenaEnvironment
from gage_eval.role.arena.sandboxed_env import SandboxedArenaEnvironment
from gage_eval.role.arena.types import ArenaAction, ArenaObservation, GameResult
from gage_eval.sandbox.base import ExecResult


class FakeSandbox:
    def __init__(self) -> None:
        self._initialized = False
        self._terminal = False
        self._move_log: list[dict[str, str]] = []
        self.init_params: dict[str, object] | None = None

    def exec(self, command: str, timeout: int = 30) -> ExecResult:  # noqa: ARG002
        request = _decode_request(command)
        method = request["method"]
        params = request.get("params") or {}
        if method == "init":
            self._initialized = True
            self.init_params = dict(params)
            return _ok({"status": "ok"})
        if method == "reset":
            self._terminal = False
            self._move_log = []
            return _ok({"status": "ok"})
        if method == "get_active_player":
            return _ok({"player_id": "p1"})
        if method == "observe":
            return _ok(
                {
                    "observation": {
                        "board_text": "...",
                        "legal_moves": ["A1", "B2"],
                        "active_player": str(params.get("player") or "p1"),
                        "metadata": {"turn": len(self._move_log)},
                    }
                }
            )
        if method == "apply":
            action = params.get("action") or {}
            self._move_log.append(
                {"player": str(action.get("player")), "move": str(action.get("move"))}
            )
            if len(self._move_log) >= 2:
                self._terminal = True
                return _ok(
                    {
                        "game_result": {
                            "winner": "p1",
                            "result": "win",
                            "reason": "line_complete",
                            "move_count": len(self._move_log),
                            "illegal_move_count": 0,
                            "final_board": "done",
                            "move_log": list(self._move_log),
                        }
                    }
                )
            return _ok({"game_result": None})
        if method == "is_terminal":
            return _ok({"terminal": self._terminal})
        if method == "build_result":
            return _ok(
                {
                    "game_result": {
                        "winner": None,
                        "result": str(params.get("result") or ""),
                        "reason": params.get("reason"),
                        "move_count": len(self._move_log),
                        "illegal_move_count": 0,
                        "final_board": "snapshot",
                        "move_log": list(self._move_log),
                    }
                }
            )
        return ExecResult(exit_code=1, stdout="", stderr=f"unknown method: {method}")


def _decode_request(command: str) -> dict:
    match = re.search(
        r"base64\.b64decode\((['\"])(?P<data>.+?)\1\)", command, re.DOTALL
    )
    assert match is not None
    payload = base64.b64decode(match.group("data")).decode("utf-8")
    return json.loads(payload)


def _ok(payload: dict) -> ExecResult:
    return ExecResult(
        exit_code=0, stdout=json.dumps(payload), stderr="", duration_ms=1.0
    )


@pytest.mark.fast
def test_sandboxed_env_satisfies_protocol() -> None:
    env = SandboxedArenaEnvironment(
        sandbox=FakeSandbox(),
        env_kwargs={"impl": "tictactoe_v1", "player_ids": ["p1", "p2"]},
    )

    assert isinstance(env, ArenaEnvironment)


@pytest.mark.fast
def test_sandboxed_env_forwards_init_and_observe() -> None:
    sandbox = FakeSandbox()
    env = SandboxedArenaEnvironment(
        sandbox=sandbox,
        env_kwargs={
            "impl": "tictactoe_v1",
            "board_size": 3,
            "player_ids": ["p1", "p2"],
            "illegal_policy": {"retry": 1},
        },
    )

    env.reset()
    observation = env.observe("p1")

    assert sandbox.init_params is not None
    assert sandbox.init_params["player_ids"] == ["p1", "p2"]
    assert isinstance(observation, ArenaObservation)
    assert observation.active_player == "p1"


@pytest.mark.fast
def test_sandboxed_env_apply_build_result_and_terminal_state() -> None:
    env = SandboxedArenaEnvironment(
        sandbox=FakeSandbox(),
        env_kwargs={"impl": "tictactoe_v1"},
    )

    env.reset()
    first = env.apply(ArenaAction(player="p1", move="A1", raw="A1"))
    second = env.apply(ArenaAction(player="p2", move="B2", raw="B2"))
    built = env.build_result(result="draw", reason="max_turns")

    assert first is None
    assert isinstance(second, GameResult)
    assert env.is_terminal() is True
    assert built.result == "draw"
