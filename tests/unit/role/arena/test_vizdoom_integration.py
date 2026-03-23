from __future__ import annotations

import base64
import contextlib
import json
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import pytest

from gage_eval.role.adapters import arena as arena_module
from gage_eval.role.adapters.arena import ArenaRoleAdapter
from gage_eval.role.adapters.human import HumanAdapter
from gage_eval.role.arena.games.vizdoom import env as vizdoom_env_module
from gage_eval.role.arena.players.human_player import HumanPlayer
from gage_eval.role.arena.players.llm_player import LLMPlayer
from gage_eval.role.arena.types import ArenaAction, ArenaObservation


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


class _QueueRole:
    def invoke(self, payload: dict[str, Any], trace: Any) -> dict[str, Any]:
        _ = trace
        queue = payload["action_queue"]
        return {"answer": queue.get()}


class _RoleManagerStub:
    def __init__(self, *, answer: str = "2", adapter: Any = None) -> None:
        self._answer = answer
        self._adapter = adapter

    @contextlib.contextmanager
    def borrow_role(self, adapter_id: str, **_kwargs) -> Iterator[_DummyRole]:
        _ = adapter_id
        yield _DummyRole(self._answer)

    def get_adapter(self, adapter_id: str) -> Any:
        _ = adapter_id
        return self._adapter


class _QueueRoleManagerStub(_RoleManagerStub):
    @contextlib.contextmanager
    def borrow_role(self, adapter_id: str) -> Iterator[_QueueRole]:
        _ = adapter_id
        yield _QueueRole()


class _FrameStub:
    def __init__(self, marker: int) -> None:
        self.shape = (1, 1, 3)
        self.dtype = "uint8"
        self._bytes = bytes([marker, 0, 0])

    def tobytes(self) -> bytes:
        return self._bytes


class _VizDoomBackendStub:
    def __init__(self) -> None:
        self._pov_frames = {0: _FrameStub(11), 1: _FrameStub(22)}
        self.view_history: list[str] = []
        self._step_count = 0

    def reset(self, seed: Optional[int] = None) -> dict[int, dict[str, Any]]:
        _ = seed
        self._step_count = 0
        self._pov_frames = {0: _FrameStub(11), 1: _FrameStub(22)}
        return {0: {"HEALTH": 100.0}, 1: {"HEALTH": 100.0}}

    def close(self) -> None:
        return None

    def get_pov_frames(self) -> dict[int, _FrameStub]:
        return dict(self._pov_frames)

    def step(self, actions: dict[int, int]) -> tuple[dict[int, dict[str, Any]], dict[int, float], bool, dict[str, Any]]:
        _ = actions
        self._step_count += 1
        self._pov_frames = {
            0: _FrameStub(11 + self._step_count),
            1: _FrameStub(22 + self._step_count),
        }
        observations = {
            0: {"HEALTH": 100.0 - self._step_count},
            1: {"HEALTH": 100.0 - self._step_count},
        }
        rewards = {0: 0.0, 1: 0.0}
        return observations, rewards, False, {}

    def set_view(self, view: str) -> None:
        self.view_history.append(view)


def _decode_raw_frame_marker(image_payload: dict[str, Any]) -> int:
    raw = base64.b64decode(str(image_payload["data"]))
    return raw[0]


def _build_vizdoom_observation() -> ArenaObservation:
    return ArenaObservation(
        board_text="",
        legal_moves=["1", "2", "3"],
        active_player="p0",
        metadata={
            "game_type": "vizdoom",
            "action_mapping": {"1": "ATTACK", "2": "TURN_LEFT", "3": "TURN_RIGHT"},
        },
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
    assert action.metadata["trace_action_applied"] == "TURN_LEFT"


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


def test_human_player_async_polls_queue_source_adapter() -> None:
    from queue import Queue

    observation = _build_vizdoom_observation()
    queue: Queue[str] = Queue()
    queue.put("2")

    adapter = HumanAdapter(adapter_id="human_queue", source="queue")
    role_manager = _RoleManagerStub(adapter=adapter)
    player = HumanPlayer(
        name="p0",
        adapter_id="human_queue",
        role_manager=role_manager,
        sample={},
        parser=_ParserStub(),
        action_queue=queue,
        timeout_ms=100,
    )

    assert player.start_thinking(observation, deadline_ms=100) is True
    for _ in range(20):
        if player.has_action():
            break
        time.sleep(0.005)

    action = player.pop_action()
    assert action.move == "2"
    assert action.metadata["player_type"] == "human"
    assert action.metadata["trace_action_applied"] == "TURN_LEFT"


def test_human_player_async_queue_payload_respects_sample_and_player_route() -> None:
    from queue import Queue

    observation = _build_vizdoom_observation()
    queue: Queue[str] = Queue()
    queue.put(
        json.dumps(
            {"sample_id": "sample-other", "player_id": "p0", "move": "1"},
            ensure_ascii=False,
        )
    )
    queue.put(
        json.dumps(
            {"sample_id": "sample-self", "player_id": "p0", "move": "2"},
            ensure_ascii=False,
        )
    )

    adapter = HumanAdapter(adapter_id="human_queue", source="queue")
    role_manager = _RoleManagerStub(adapter=adapter)
    player = HumanPlayer(
        name="p0",
        adapter_id="human_queue",
        role_manager=role_manager,
        sample={"id": "sample-self"},
        parser=_ParserStub(),
        action_queue=queue,
        timeout_ms=100,
    )

    assert player.start_thinking(observation, deadline_ms=100) is True
    for _ in range(20):
        if player.has_action():
            break
        time.sleep(0.005)

    action = player.pop_action()
    assert action.move == "2"
    assert action.metadata["player_type"] == "human"

    requeued = json.loads(queue.get_nowait())
    assert requeued["sample_id"] == "sample-other"
    assert requeued["player_id"] == "p0"
    assert requeued["move"] == "1"


def test_human_player_sync_queue_payload_waits_for_matching_sample_route() -> None:
    from queue import Queue

    observation = _build_vizdoom_observation()
    queue: Queue[str] = Queue()
    queue.put(
        json.dumps(
            {"sample_id": "sample-other", "player_id": "p0", "move": "1"},
            ensure_ascii=False,
        )
    )
    queue.put(
        json.dumps(
            {"sample_id": "sample-sync", "player_id": "p0", "move": "2"},
            ensure_ascii=False,
        )
    )

    player = HumanPlayer(
        name="p0",
        adapter_id="human_queue",
        role_manager=_QueueRoleManagerStub(),
        sample={"id": "sample-sync"},
        parser=_ParserStub(),
        action_queue=queue,
    )

    action = player.think(observation)

    assert action.move == "2"
    assert action.metadata["player_type"] == "human"

    requeued = json.loads(queue.get_nowait())
    assert requeued["sample_id"] == "sample-other"
    assert requeued["player_id"] == "p0"
    assert requeued["move"] == "1"


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
    assert action.metadata["trace_action_applied"] == "TURN_LEFT"


def test_vizdoom_observe_includes_action_mapping_in_metadata(monkeypatch) -> None:
    backend = _VizDoomBackendStub()
    monkeypatch.setattr(
        vizdoom_env_module.ViZDoomArenaEnvironment,
        "_build_env",
        lambda self, cfg: backend,
    )
    env = vizdoom_env_module.ViZDoomArenaEnvironment(
        player_ids=["p0", "p1"],
        show_pov=False,
        capture_pov=True,
        replay_in_env=False,
    )
    env.reset()

    observation = env.observe("p0")

    assert observation.metadata["action_mapping"] == {
        "1": "ATTACK",
        "2": "TURN_LEFT",
        "3": "TURN_RIGHT",
    }


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
            "obs_image_history_len": 4,
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
    assert captured_env_kwargs["obs_image_history_len"] == 4
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


def test_vizdoom_observe_uses_player_specific_pov_frame(monkeypatch) -> None:
    backend = _VizDoomBackendStub()
    monkeypatch.setattr(
        vizdoom_env_module.ViZDoomArenaEnvironment,
        "_build_env",
        lambda self, cfg: backend,
    )
    env = vizdoom_env_module.ViZDoomArenaEnvironment(
        player_ids=["p0", "p1"],
        show_pov=False,
        capture_pov=True,
        obs_image=True,
        replay_in_env=False,
    )
    env.reset()

    p0_obs = env.observe("p0")
    p1_obs = env.observe("p1")

    assert _decode_raw_frame_marker(p0_obs.view["image"]) == 11
    assert _decode_raw_frame_marker(p1_obs.view["image"]) == 22


def test_vizdoom_last_frame_respects_configured_pov_view(monkeypatch) -> None:
    backend = _VizDoomBackendStub()
    monkeypatch.setattr(
        vizdoom_env_module.ViZDoomArenaEnvironment,
        "_build_env",
        lambda self, cfg: backend,
    )
    env = vizdoom_env_module.ViZDoomArenaEnvironment(
        player_ids=["p0", "p1"],
        show_pov=False,
        capture_pov=True,
        pov_view="p1",
        replay_in_env=False,
    )
    env.reset()

    frame_payload = env.get_last_frame()

    assert frame_payload["player_index"] == 1
    assert frame_payload["actor"] == "p1"

    env.set_view("p0")
    frame_payload = env.get_last_frame()

    assert backend.view_history[-1] == "p0"
    assert frame_payload["player_index"] == 0
    assert frame_payload["actor"] == "p0"


def test_vizdoom_observe_exposes_ordered_image_history(monkeypatch) -> None:
    backend = _VizDoomBackendStub()
    monkeypatch.setattr(
        vizdoom_env_module.ViZDoomArenaEnvironment,
        "_build_env",
        lambda self, cfg: backend,
    )
    env = vizdoom_env_module.ViZDoomArenaEnvironment(
        player_ids=["p0", "p1"],
        show_pov=False,
        capture_pov=True,
        obs_image=True,
        obs_image_history_len=4,
        replay_in_env=False,
    )
    env.reset()

    for _ in range(3):
        result = env.apply(
            ArenaAction(
                player="system",
                move={"p0": "1", "p1": "2"},
                raw="batch",
            )
        )
        assert result is None

    p1_obs = env.observe("p1")

    markers = [_decode_raw_frame_marker(frame) for frame in p1_obs.view["image_history"]]
    assert markers == [22, 23, 24, 25]
    assert _decode_raw_frame_marker(p1_obs.view["image"]) == 25
