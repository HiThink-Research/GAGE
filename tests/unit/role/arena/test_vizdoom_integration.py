from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import pytest

from gage_eval.role.adapters import arena as arena_module
from gage_eval.role.adapters.arena import ArenaRoleAdapter
from gage_eval.role.arena.games.vizdoom import env as vizdoom_env_module
from gage_eval.role.arena.players.human_player import HumanPlayer
from gage_eval.role.arena.players.llm_player import LLMPlayer
from gage_eval.role.arena.types import ArenaObservation


@dataclass
class _ParseResult:
    coord: Optional[str]
    error: Optional[str]
    chat_text: Optional[str] = None


class _ParserStub:
    def parse(self, text: str, *, legal_moves=None) -> _ParseResult:
        _ = legal_moves
        return _ParseResult(coord=str(text).strip(), error=None)

    def build_rethink_prompt(
        self,
        *,
        last_output: str,
        reason: str,
        legal_moves: list[str],
    ) -> str:
        return f"retry: {last_output} {reason} {legal_moves}"


class _PollingAdapter:
    def __init__(self, responses: list[Optional[str]]) -> None:
        self._responses = list(responses)
        self.ready_called = False

    def ensure_input_ready(self) -> None:
        self.ready_called = True

    def poll_action(
        self, *, timeout_ms: Optional[int] = None, default_action: Optional[str] = None
    ) -> Optional[str]:
        _ = timeout_ms, default_action
        if self._responses:
            return self._responses.pop(0)
        return None


class _DummyRole:
    def __init__(self, answer: str) -> None:
        self._answer = answer

    def invoke(self, payload: dict[str, Any], trace: Any) -> dict[str, str]:
        _ = payload, trace
        return {"answer": self._answer}


class _RoleManagerStub:
    def __init__(self, *, answer: str = "2", adapter: Any = None) -> None:
        self._answer = answer
        self._adapter = adapter

    @contextlib.contextmanager
    def borrow_role(self, adapter_id: str) -> Iterator[_DummyRole]:
        _ = adapter_id
        yield _DummyRole(self._answer)

    def get_adapter(self, adapter_id: str) -> Any:
        _ = adapter_id
        return self._adapter


def _build_vizdoom_observation() -> ArenaObservation:
    return ArenaObservation(
        board_text="",
        legal_moves=["1", "2", "3"],
        active_player="p0",
        metadata={"game_type": "vizdoom"},
        view={"text": "tick 0"},
        legal_actions={"items": ["1", "2", "3"]},
    )


def test_human_player_async_polls_action() -> None:
    observation = _build_vizdoom_observation()
    adapter = _PollingAdapter(["2"])
    role_manager = _RoleManagerStub(adapter=adapter)
    player = HumanPlayer(
        name="p0",
        adapter_id="human",
        role_manager=role_manager,
        sample={},
        parser=_ParserStub(),
        timeout_ms=100,
    )

    assert player.start_thinking(observation, deadline_ms=100) is True
    assert adapter.ready_called is True

    for _ in range(20):
        if player.has_action():
            break
        time.sleep(0.005)

    action = player.pop_action()
    assert action.move == "2"
    assert action.metadata["player_type"] == "human"


def test_human_player_async_uses_timeout_fallback() -> None:
    observation = _build_vizdoom_observation()
    adapter = _PollingAdapter([None, None, None])
    role_manager = _RoleManagerStub(adapter=adapter)
    player = HumanPlayer(
        name="p0",
        adapter_id="human",
        role_manager=role_manager,
        sample={},
        parser=_ParserStub(),
        timeout_ms=1,
        timeout_fallback_move="0",
    )

    assert player.start_thinking(observation, deadline_ms=1) is True
    time.sleep(0.02)
    assert player.has_action() is True

    action = player.pop_action()
    assert action.move == "0"
    assert action.metadata["fallback"] == "timeout_noop"


def test_llm_player_async_returns_action() -> None:
    observation = _build_vizdoom_observation()
    role_manager = _RoleManagerStub(answer="2")
    player = LLMPlayer(
        name="p1",
        adapter_id="backend",
        role_manager=role_manager,
        sample={"messages": []},
        parser=_ParserStub(),
        timeout_ms=100,
    )

    assert player.start_thinking(observation, deadline_ms=100) is True

    for _ in range(100):
        if player.has_action():
            break
        time.sleep(0.01)

    action = player.pop_action()
    assert action.move == "2"
    assert action.metadata["player_type"] == "backend"


def test_arena_adapter_forwards_vizdoom_env_kwargs(monkeypatch) -> None:
    captured_env_kwargs: dict[str, Any] = {}

    class _CapturedEnv:
        def __init__(self, **kwargs: Any) -> None:
            captured_env_kwargs.update(kwargs)

    monkeypatch.setattr(arena_module.registry, "get", lambda kind, impl: _CapturedEnv)
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={
            "impl": "vizdoom_env_v1",
            "render_mode": "both",
            "show_automap": False,
            "show_pov": True,
            "max_steps": 123,
            "action_repeat": 2,
            "allow_partial_actions": True,
            "reset_retry_count": 4,
            "death_check_warmup_steps": 6,
        },
    )

    adapter._build_environment(
        {"metadata": {}},
        player_ids=["p0", "p1"],
        player_names={"p0": "P0", "p1": "P1"},
    )

    assert captured_env_kwargs["render_mode"] == "both"
    assert captured_env_kwargs["show_automap"] is False
    assert captured_env_kwargs["show_pov"] is True
    assert captured_env_kwargs["max_steps"] == 123
    assert captured_env_kwargs["action_repeat"] == 2
    assert captured_env_kwargs["allow_partial_actions"] is True
    assert captured_env_kwargs["reset_retry_count"] == 4
    assert captured_env_kwargs["death_check_warmup_steps"] == 6


def test_arena_adapter_enables_capture_pov_when_replay_mode_includes_frame(monkeypatch) -> None:
    captured_env_kwargs: dict[str, Any] = {}

    class _CapturedEnv:
        def __init__(self, **kwargs: Any) -> None:
            captured_env_kwargs.update(kwargs)

    monkeypatch.setattr(arena_module.registry, "get", lambda kind, impl: _CapturedEnv)
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={
            "impl": "vizdoom_env_v1",
            "show_pov": False,
            "replay": {"enabled": True, "mode": "both"},
        },
    )

    adapter._build_environment(
        {"metadata": {}},
        player_ids=["p0", "p1"],
        player_names={"p0": "P0", "p1": "P1"},
    )

    assert captured_env_kwargs["show_pov"] is False
    assert captured_env_kwargs["capture_pov"] is True


def test_arena_adapter_rejects_subprocess_player() -> None:
    adapter = ArenaRoleAdapter(adapter_id="arena")
    with pytest.raises(ValueError, match="Unsupported player type: subprocess"):
        adapter._build_players(
            sample={},
            role_manager=object(),
            parser=_ParserStub(),
            trace=None,
            action_queue=None,
            player_specs=[
                {
                    "player_id": "p0",
                    "type": "subprocess",
                    "cmd": ["python", "bot.py"],
                }
            ],
        )


def test_vizdoom_backend_loader_wraps_import_error(monkeypatch) -> None:
    def _raise_import_error(module_name: str) -> Any:
        raise ImportError(f"missing module: {module_name}")

    monkeypatch.setattr(vizdoom_env_module.importlib, "import_module", _raise_import_error)

    with pytest.raises(ValueError, match="optional deps"):
        vizdoom_env_module._load_vizdoom_backend_module("dummy.module")
