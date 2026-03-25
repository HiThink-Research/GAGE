from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from uuid import uuid4

from gage_eval.registry import registry
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder
from gage_eval.role.adapters import arena as arena_module
from gage_eval.role.adapters.arena import ArenaRoleAdapter, _VisualizedEnvironment
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.arena.sandboxed_env import SandboxedArenaEnvironment
from gage_eval.role.arena.visualizers import gradio_visualizer as gradio_visualizer_module
from gage_eval.role.arena.games.common.grid_coord_input_mapper import (
    GridCoordInputMapper,
)
from gage_eval.role.arena.games.doudizhu.doudizhu_input_mapper import (
    DoudizhuInputMapper,
)
from gage_eval.role.arena.games.mahjong.mahjong_input_mapper import MahjongInputMapper
from gage_eval.role.arena.games.pettingzoo.pettingzoo_input_mapper import PettingZooDiscreteInputMapper
from gage_eval.role.arena.games.vizdoom.vizdoom_input_mapper import ViZDoomInputMapper
from gage_eval.role.arena.players.llm_player import LLMPlayer
from gage_eval.role.arena.schedulers.turn_scheduler import TurnScheduler
from gage_eval.role.arena.types import ArenaObservation, GameResult
from gage_eval.tools import action_server as action_server_module


def _make_result(
    *,
    move_log: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    replay_path: str | None = None,
    arena_trace: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> GameResult:
    return GameResult(
        winner=None,
        result="draw",
        reason=None,
        move_count=len(move_log),
        illegal_move_count=0,
        final_board="",
        move_log=move_log,
        replay_path=replay_path,
        arena_trace=arena_trace,
    )


def test_ainvoke_runs_sync_path_in_executor(monkeypatch) -> None:
    adapter = ArenaRoleAdapter(adapter_id="arena")
    monkeypatch.setattr(
        adapter, "_invoke_sync", lambda payload, state: {"ok": True, "payload": payload}
    )

    result = asyncio.run(adapter.ainvoke({"sample_id": "s1"}, RoleAdapterState()))

    assert result == {"ok": True, "payload": {"sample_id": "s1"}}


def test_build_scheduler_supports_turn_type() -> None:
    adapter = ArenaRoleAdapter(adapter_id="arena", scheduler={"type": " TURN "})

    scheduler = adapter._build_scheduler({"eval_config": {"max_turns": 7}})

    assert isinstance(scheduler, TurnScheduler)


def test_build_parser_prefers_runtime_registry_view(monkeypatch) -> None:
    impl_name = f"test_parser_{uuid4().hex}"
    clone = registry.clone()

    class _Parser:
        def __init__(self, *, board_size: int, coord_scheme: str = "A1") -> None:
            self.board_size = board_size
            self.coord_scheme = coord_scheme

        def parse(self, text: str, *, legal_moves=None) -> dict[str, Any]:
            return {"text": text, "legal_moves": legal_moves}

        def build_rethink_prompt(
            self,
            *,
            last_output: str,
            reason: str,
            legal_moves: list[str],
        ) -> str:
            return f"{last_output}|{reason}|{legal_moves}"

    clone.register("parser_impls", impl_name, _Parser, desc="test parser")
    view = clone.freeze(view_id=f"parser-view-{uuid4().hex}")
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        parser={"impl": impl_name, "board_size": 9, "coord_scheme": "ROW_COL"},
        registry_view=view,
    )

    monkeypatch.setattr(
        arena_module,
        "import_arena_asset_module",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("runtime registry view must not trigger manifest fallback")
        ),
    )

    parser = adapter._build_parser({"metadata": {"board_size": 7}})

    assert parser.board_size == 7
    assert parser.coord_scheme == "ROW_COL"


def test_build_environment_prefers_runtime_registry_view(monkeypatch) -> None:
    impl_name = f"test_env_{uuid4().hex}"
    clone = registry.clone()

    class _Environment:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _Provider:
        def enrich_environment_kwargs(self, **kwargs: Any) -> dict[str, Any]:
            return dict(kwargs["env_kwargs"])

    clone.register("arena_impls", impl_name, _Environment, desc="test environment")
    view = clone.freeze(view_id=f"env-view-{uuid4().hex}")
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": impl_name},
        registry_view=view,
    )

    monkeypatch.setattr(adapter, "_resolve_game_provider", lambda env_impl: _Provider())
    monkeypatch.setattr(
        arena_module,
        "import_arena_asset_module",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("runtime registry view must not trigger manifest fallback")
        ),
    )

    environment = adapter._build_environment({"metadata": {}})

    assert isinstance(environment, _Environment)
    assert environment.kwargs["board_size"] == 15


def test_normalize_player_specs_prefers_adapter_ref_for_generic_name() -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        players=[
            {"id": "p0", "type": "backend", "ref": "backend.alpha"},
        ],
    )
    sample = {
        "metadata": {
            "player_ids": ["p0"],
            "player_names": {"p0": "player 0"},
        }
    }

    _, _, player_names, _ = adapter._normalize_player_specs(sample)

    assert player_names["p0"] == "backend.alpha"


def test_invoke_sync_mahjong_passes_resolved_player_models(monkeypatch) -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "mahjong_mock_impl"},
        players=[{"player_id": "p0", "type": "backend", "ref": "backend.alpha"}],
    )
    captured_env_kwargs: dict[str, Any] = {}

    class _StubScheduler:
        def run_loop(self, environment, players) -> GameResult:
            return _make_result(move_log=[])

    monkeypatch.setattr(
        adapter,
        "_normalize_player_specs",
        lambda sample: (
            [{"player_id": "p0", "type": "backend", "ref": "backend.alpha"}],
            ["p0"],
            {"p0": "P0"},
            "p0",
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_resolve_player_labels",
        lambda specs, role_manager: {"p0": "model-001"},
    )
    monkeypatch.setattr(adapter, "_build_parser", lambda sample: object())
    monkeypatch.setattr(adapter, "_build_scheduler", lambda sample: _StubScheduler())
    monkeypatch.setattr(
        adapter, "_ensure_visualizer", lambda sample, player_specs: (None, None)
    )
    monkeypatch.setattr(
        adapter, "_ensure_action_server", lambda player_specs: (None, None)
    )
    monkeypatch.setattr(
        adapter,
        "_build_environment",
        lambda sample, **kwargs: captured_env_kwargs.update(kwargs) or object(),
    )
    monkeypatch.setattr(adapter, "_build_players", lambda *args, **kwargs: [object()])

    output = adapter._invoke_sync(
        {"sample": {"metadata": {}}, "role_manager": object()}, RoleAdapterState()
    )

    assert output["result"] == "draw"
    assert captured_env_kwargs["player_models"] == {"p0": "model-001"}


def test_invoke_sync_doudizhu_fills_missing_player_name_from_label(monkeypatch) -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "doudizhu_mock_impl"},
        players=[{"player_id": "p0", "type": "backend", "ref": "backend.alpha"}],
    )
    captured_env_kwargs: dict[str, Any] = {}

    class _StubScheduler:
        def run_loop(self, environment, players) -> GameResult:
            return _make_result(move_log=[])

    monkeypatch.setattr(
        adapter,
        "_normalize_player_specs",
        lambda sample: (
            [{"player_id": "p0", "type": "backend", "ref": "backend.alpha"}],
            ["p0"],
            {},
            "p0",
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_resolve_player_labels",
        lambda specs, role_manager: {"p0": "model-002"},
    )
    monkeypatch.setattr(adapter, "_build_parser", lambda sample: object())
    monkeypatch.setattr(adapter, "_build_scheduler", lambda sample: _StubScheduler())
    monkeypatch.setattr(
        adapter, "_ensure_visualizer", lambda sample, player_specs: (None, None)
    )
    monkeypatch.setattr(
        adapter, "_ensure_action_server", lambda player_specs: (None, None)
    )
    monkeypatch.setattr(
        adapter,
        "_build_environment",
        lambda sample, **kwargs: captured_env_kwargs.update(kwargs) or object(),
    )
    monkeypatch.setattr(adapter, "_build_players", lambda *args, **kwargs: [object()])

    adapter._invoke_sync(
        {"sample": {"metadata": {}}, "role_manager": object()}, RoleAdapterState()
    )

    assert captured_env_kwargs["player_names"]["p0"] == "model-002"


def test_invoke_sync_waits_for_pending_players(monkeypatch) -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        scheduler={"pending_wait_timeout_s": 1.5},
    )

    class _StubScheduler:
        def run_loop(self, environment, players) -> GameResult:
            _ = environment, players
            return _make_result(move_log=[])

    class _PendingPlayer:
        def __init__(self) -> None:
            self.name = "p0"
            self.wait_calls = 0
            self.last_timeout_s: float = 0.0

        def wait_for_pending(self, timeout_s: float = 1.0) -> None:
            self.wait_calls += 1
            self.last_timeout_s = float(timeout_s)

    pending_player = _PendingPlayer()
    monkeypatch.setattr(
        adapter,
        "_normalize_player_specs",
        lambda sample: (
            [{"player_id": "p0", "type": "backend", "ref": "backend.alpha"}],
            ["p0"],
            {"p0": "P0"},
            "p0",
        ),
    )
    monkeypatch.setattr(adapter, "_build_parser", lambda sample: object())
    monkeypatch.setattr(adapter, "_build_scheduler", lambda sample: _StubScheduler())
    monkeypatch.setattr(
        adapter, "_ensure_visualizer", lambda sample, player_specs: (None, None)
    )
    monkeypatch.setattr(
        adapter, "_ensure_action_server", lambda player_specs: (None, None)
    )
    monkeypatch.setattr(
        adapter, "_build_environment", lambda sample, **kwargs: object()
    )
    monkeypatch.setattr(
        adapter, "_build_players", lambda *args, **kwargs: [pending_player]
    )

    output = adapter._invoke_sync(
        {"sample": {"metadata": {}}, "role_manager": object()}, RoleAdapterState()
    )

    assert output["result"] == "draw"
    assert pending_player.wait_calls == 1
    assert 0.0 < pending_player.last_timeout_s <= 1.5


def test_invoke_sync_emits_loop_lifecycle_events(monkeypatch) -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        players=[{"player_id": "p0", "type": "backend", "ref": "backend.alpha"}],
    )
    trace = ObservabilityTrace(
        recorder=InMemoryRecorder(run_id="arena-loop-events"),
        run_id="arena-loop-events",
    )

    class _StubScheduler:
        @staticmethod
        def run_loop(environment, players) -> GameResult:
            _ = environment, players
            return _make_result(move_log=[{"move": "A1"}])

    monkeypatch.setattr(
        adapter,
        "_normalize_player_specs",
        lambda sample: (
            [{"player_id": "p0", "type": "backend", "ref": "backend.alpha"}],
            ["p0"],
            {"p0": "P0"},
            "p0",
        ),
    )
    monkeypatch.setattr(adapter, "_build_parser", lambda sample: object())
    monkeypatch.setattr(adapter, "_build_scheduler", lambda sample: _StubScheduler())
    monkeypatch.setattr(
        adapter, "_ensure_visualizer", lambda sample, player_specs: (None, None)
    )
    monkeypatch.setattr(
        adapter, "_ensure_action_server", lambda player_specs: (None, None)
    )
    monkeypatch.setattr(
        adapter,
        "_build_environment",
        lambda sample, **kwargs: object(),
    )
    monkeypatch.setattr(adapter, "_build_players", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(
        adapter,
        "_format_result",
        lambda result, sample, trace, frame_events=None: {"result": result.result},
    )

    output = adapter._invoke_sync(
        {"sample": {"metadata": {}}, "role_manager": object(), "trace": trace},
        RoleAdapterState(),
    )

    assert output["result"] == "draw"
    events = [event["event"] for event in trace.events]
    assert "arena_loop_start" in events
    assert "arena_loop_end" in events
    assert "arena_start" not in events
    assert "arena_end" not in events


def test_build_environment_returns_sandboxed_env_when_provider_present() -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "tictactoe_v1", "board_size": 3},
        rules={"win_len": 3},
    )

    class _FakeSandbox:
        def exec(self, command: str, timeout: int = 30):  # noqa: ARG002
            raise AssertionError(
                "sandbox exec should not be called during construction"
            )

    class _FakeHandle:
        sandbox = _FakeSandbox()

    class _FakeProvider:
        def get_handle(self):
            return _FakeHandle()

    environment = adapter._build_environment(
        {"metadata": {}},
        player_ids=["p1", "p2"],
        player_names={"p1": "Alice", "p2": "Bob"},
        start_player_id="p1",
        sandbox_provider=_FakeProvider(),
    )

    assert isinstance(environment, SandboxedArenaEnvironment)


def test_build_environment_for_pettingzoo_forwards_optional_kwargs(monkeypatch) -> None:
    captured_env_kwargs: dict[str, Any] = {}

    class _CapturedEnv:
        def __init__(self, **kwargs: Any) -> None:
            captured_env_kwargs.update(kwargs)

    monkeypatch.setattr(arena_module.registry, "get", lambda kind, impl: _CapturedEnv)
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={
            "impl": "pettingzoo_mock_impl",
            "env_id": "pong_v3",
            "env_kwargs": {"frameskip": 4},
            "seed": 123,
            "action_labels": {"0": "noop"},
            "use_action_meanings": True,
            "include_raw_obs": True,
            "agent_map": {"first_0": "p0"},
        },
    )

    adapter._build_environment(
        {"metadata": {}},
        player_ids=["p0"],
        player_names={"p0": "player0"},
    )

    assert captured_env_kwargs["env_id"] == "pong_v3"
    assert captured_env_kwargs["env_kwargs"] == {"frameskip": 4}
    assert captured_env_kwargs["seed"] == 123
    assert captured_env_kwargs["action_labels"] == {"0": "noop"}
    assert captured_env_kwargs["use_action_meanings"] is True
    assert captured_env_kwargs["include_raw_obs"] is True
    assert captured_env_kwargs["agent_map"] == {"first_0": "p0"}


def test_build_environment_for_gomoku_forwards_obs_image(monkeypatch) -> None:
    captured_env_kwargs: dict[str, Any] = {}

    class _CapturedEnv:
        def __init__(self, **kwargs: Any) -> None:
            captured_env_kwargs.update(kwargs)

    monkeypatch.setattr(arena_module.registry, "get", lambda kind, impl: _CapturedEnv)
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={
            "impl": "gomoku_local_v1",
            "board_size": 15,
            "coord_scheme": "A1",
            "obs_image": True,
        },
    )

    adapter._build_environment(
        {"metadata": {}},
        player_ids=["Black", "White"],
        player_names={"Black": "Black", "White": "White"},
    )

    assert captured_env_kwargs["obs_image"] is True


def test_build_players_forwards_scheme_configuration() -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        players=[
            {
                "player_id": "p0",
                "type": "backend",
                "ref": "backend.alpha",
                "scheme_id": "S6_text_image_action_hist",
                "scheme_params": {"action_history_len": 3, "delta_key_limit": 5},
            }
        ],
    )

    players = adapter._build_players(
        sample={"messages": [], "metadata": {"game_type": "vizdoom"}},
        role_manager=object(),
        parser=object(),
        trace=None,
        action_queue=None,
        player_specs=adapter._player_specs,
    )

    assert len(players) == 1
    assert isinstance(players[0], LLMPlayer)
    assert players[0]._scheme_id == "S6_text_image_action_hist"
    assert players[0]._action_history_len == 3
    assert players[0]._delta_key_limit == 5


def test_build_environment_for_mahjong_forwards_run_context_and_models(monkeypatch) -> None:
    captured_env_kwargs: dict[str, Any] = {}
    chat_queue = object()

    class _CapturedEnv:
        def __init__(self, **kwargs: Any) -> None:
            captured_env_kwargs.update(kwargs)

    monkeypatch.setattr(arena_module.registry, "get", lambda kind, impl: _CapturedEnv)
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={
            "impl": "mahjong_mock_impl",
            "chat_every_n": 3,
            "replay_live": True,
            "replay_output_dir": "replays",
            "replay_filename": "sample.json",
        },
    )
    trace = ObservabilityTrace(run_id="run-xyz")

    adapter._build_environment(
        {"id": "sample-7", "metadata": {}},
        player_ids=["p0"],
        player_names={"p0": "player0"},
        player_models={"p0": "model-007"},
        chat_queue=chat_queue,
        trace=trace,
    )

    assert captured_env_kwargs["run_id"] == "run-xyz"
    assert captured_env_kwargs["sample_id"] == "sample-7"
    assert captured_env_kwargs["chat_every_n"] == 3
    assert captured_env_kwargs["replay_live"] is True
    assert captured_env_kwargs["replay_output_dir"] == "replays"
    assert captured_env_kwargs["replay_filename"] == "sample.json"
    assert captured_env_kwargs["chat_queue"] is chat_queue
    assert captured_env_kwargs["player_models"] == {"p0": "model-007"}


def test_build_environment_for_retro_forwards_runtime_and_observation_kwargs(
    monkeypatch,
) -> None:
    captured_env_kwargs: dict[str, Any] = {}

    class _CapturedEnv:
        def __init__(self, **kwargs: Any) -> None:
            captured_env_kwargs.update(kwargs)

    monkeypatch.setattr(arena_module.registry, "get", lambda kind, impl: _CapturedEnv)
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={
            "impl": "retro_env_v1",
            "game": "SuperMarioBros3-Nes-v0",
            "state": "Start",
            "display_mode": "websocket",
            "legal_moves": ["noop", "right"],
            "action_schema": {"hold_ticks_default": 6},
            "token_budget": 128,
            "frame_stride": 2,
            "snapshot_stride": 2,
            "obs_image": True,
            "record_bk2": False,
        },
    )
    trace = ObservabilityTrace(run_id="run-retro")

    adapter._build_environment(
        {"id": "sample-retro", "metadata": {}},
        player_ids=["p0"],
        player_names={"p0": "player0"},
        trace=trace,
    )

    assert captured_env_kwargs["game"] == "SuperMarioBros3-Nes-v0"
    assert captured_env_kwargs["state"] == "Start"
    assert captured_env_kwargs["display_mode"] == "websocket"
    assert captured_env_kwargs["legal_moves"] == ["noop", "right"]
    assert captured_env_kwargs["action_schema"] == {"hold_ticks_default": 6}
    assert captured_env_kwargs["token_budget"] == 128
    assert captured_env_kwargs["frame_stride"] == 2
    assert captured_env_kwargs["snapshot_stride"] == 2
    assert captured_env_kwargs["obs_image"] is True
    assert captured_env_kwargs["run_id"] == "run-retro"
    assert captured_env_kwargs["sample_id"] == "sample-retro"


def test_bind_input_mapper_returns_mahjong_mapper() -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "mahjong_rlcard_v1"},
    )

    mapper = adapter._bind_input_mapper(env_impl="mahjong_rlcard_v1")

    assert isinstance(mapper, MahjongInputMapper)


def test_bind_input_mapper_returns_doudizhu_mapper() -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "doudizhu_arena_v1"},
    )

    mapper = adapter._bind_input_mapper(env_impl="doudizhu_arena_v1")

    assert isinstance(mapper, DoudizhuInputMapper)


def test_bind_input_mapper_returns_grid_mapper_for_gomoku() -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "gomoku_local_v1", "coord_scheme": "A1"},
    )

    mapper = adapter._bind_input_mapper(env_impl="gomoku_local_v1")

    assert isinstance(mapper, GridCoordInputMapper)


def test_bind_input_mapper_returns_grid_mapper_for_tictactoe() -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "tictactoe_v1", "coord_scheme": "ROW_COL"},
    )

    mapper = adapter._bind_input_mapper(env_impl="tictactoe_v1")

    assert isinstance(mapper, GridCoordInputMapper)


def test_bind_input_mapper_returns_pettingzoo_mapper() -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "pettingzoo_aec_v1"},
    )

    mapper = adapter._bind_input_mapper(env_impl="pettingzoo_aec_v1")

    assert isinstance(mapper, PettingZooDiscreteInputMapper)


def test_bind_input_mapper_returns_vizdoom_mapper() -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "vizdoom_env_v1"},
    )

    mapper = adapter._bind_input_mapper(env_impl="vizdoom_env_v1")

    assert isinstance(mapper, ViZDoomInputMapper)


def test_maybe_register_ws_display_for_mahjong() -> None:
    class _Hub:
        def __init__(self) -> None:
            self.registrations: list[Any] = []

        def register_display(self, registration: Any) -> None:
            self.registrations.append(registration)

    class _Environment:
        @staticmethod
        def get_last_frame() -> dict[str, Any]:
            return {"frame_id": 1}

    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "mahjong_rlcard_v1", "display_mode": "websocket"},
    )
    hub = _Hub()
    adapter._ensure_ws_rgb_hub = lambda: hub  # type: ignore[method-assign]

    adapter._maybe_register_ws_display(
        sample={"id": "sample-1", "task_id": "task-1", "metadata": {}},
        environment=_Environment(),
        action_queue=None,
        player_specs=[{"type": "human", "player_id": "player_0"}],
        env_impl="mahjong_rlcard_v1",
    )

    assert len(hub.registrations) == 1
    registration = hub.registrations[0]
    assert registration.display_id == "task-1:sample-1:arena:mahjong_rlcard_v1"
    assert isinstance(registration.input_mapper, MahjongInputMapper)


def test_maybe_register_ws_display_for_doudizhu() -> None:
    class _Hub:
        def __init__(self) -> None:
            self.registrations: list[Any] = []

        def register_display(self, registration: Any) -> None:
            self.registrations.append(registration)

    class _Environment:
        @staticmethod
        def get_last_frame() -> dict[str, Any]:
            return {"frame_id": 2}

    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "doudizhu_arena_v1", "display_mode": "ws"},
    )
    hub = _Hub()
    adapter._ensure_ws_rgb_hub = lambda: hub  # type: ignore[method-assign]

    adapter._maybe_register_ws_display(
        sample={"id": "sample-2", "task_id": "task-2", "metadata": {}},
        environment=_Environment(),
        action_queue=None,
        player_specs=[{"type": "human", "player_id": "player_1"}],
        env_impl="doudizhu_arena_v1",
    )

    assert len(hub.registrations) == 1
    registration = hub.registrations[0]
    assert registration.display_id == "task-2:sample-2:arena:doudizhu_arena_v1"
    assert isinstance(registration.input_mapper, DoudizhuInputMapper)


def test_maybe_register_ws_display_for_gomoku() -> None:
    class _Hub:
        def __init__(self) -> None:
            self.registrations: list[Any] = []

        def register_display(self, registration: Any) -> None:
            self.registrations.append(registration)

    class _Environment:
        @staticmethod
        def get_last_frame() -> dict[str, Any]:
            return {"frame_id": 3}

    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "gomoku_local_v1", "display_mode": "websocket"},
    )
    hub = _Hub()
    adapter._ensure_ws_rgb_hub = lambda: hub  # type: ignore[method-assign]

    adapter._maybe_register_ws_display(
        sample={"id": "sample-3", "task_id": "task-3", "metadata": {}},
        environment=_Environment(),
        action_queue=None,
        player_specs=[{"type": "human", "player_id": "player_0"}],
        env_impl="gomoku_local_v1",
    )

    assert len(hub.registrations) == 1
    registration = hub.registrations[0]
    assert registration.display_id == "task-3:sample-3:arena:gomoku_local_v1"
    assert isinstance(registration.input_mapper, GridCoordInputMapper)


def test_maybe_register_ws_display_for_pettingzoo() -> None:
    class _Hub:
        def __init__(self) -> None:
            self.registrations: list[Any] = []

        def register_display(self, registration: Any) -> None:
            self.registrations.append(registration)

    class _Environment:
        @staticmethod
        def get_last_frame() -> dict[str, Any]:
            return {"frame_id": 4}

    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "pettingzoo_aec_v1", "display_mode": "websocket"},
    )
    hub = _Hub()
    adapter._ensure_ws_rgb_hub = lambda: hub  # type: ignore[method-assign]

    adapter._maybe_register_ws_display(
        sample={"id": "sample-4", "task_id": "task-4", "metadata": {}},
        environment=_Environment(),
        action_queue=None,
        player_specs=[{"type": "human", "player_id": "player_0"}],
        env_impl="pettingzoo_aec_v1",
    )

    assert len(hub.registrations) == 1
    registration = hub.registrations[0]
    assert registration.display_id == "task-4:sample-4:arena:pettingzoo_aec_v1"
    assert isinstance(registration.input_mapper, PettingZooDiscreteInputMapper)


def test_maybe_register_ws_display_for_vizdoom() -> None:
    class _Hub:
        def __init__(self) -> None:
            self.registrations: list[Any] = []

        def register_display(self, registration: Any) -> None:
            self.registrations.append(registration)

    class _Environment:
        @staticmethod
        def get_last_frame() -> dict[str, Any]:
            return {"frame_id": 5}

    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "vizdoom_env_v1", "display_mode": "websocket"},
    )
    hub = _Hub()
    adapter._ensure_ws_rgb_hub = lambda: hub  # type: ignore[method-assign]

    adapter._maybe_register_ws_display(
        sample={"id": "sample-5", "task_id": "task-5", "metadata": {}},
        environment=_Environment(),
        action_queue=None,
        player_specs=[{"type": "human", "player_id": "p0"}],
        env_impl="vizdoom_env_v1",
    )

    assert len(hub.registrations) == 1
    registration = hub.registrations[0]
    assert registration.display_id == "task-5:sample-5:arena:vizdoom_env_v1"
    assert isinstance(registration.input_mapper, ViZDoomInputMapper)
    assert callable(registration.session_state_source)
    assert callable(registration.terminate_game)
    assert callable(registration.terminate_process)


def test_session_controlled_environment_manual_stop_finalizes_result() -> None:
    class _BaseEnv:
        def __init__(self) -> None:
            self.finalize_calls = 0

        @staticmethod
        def reset() -> None:
            return None

        @staticmethod
        def get_active_player() -> str:
            return "player_0"

        @staticmethod
        def observe(player: str) -> ArenaObservation:
            return ArenaObservation(
                board_text="board",
                legal_moves=["0"],
                active_player=player,
            )

        @staticmethod
        def get_last_frame() -> dict[str, Any]:
            return {"board_text": "frame-1"}

        @staticmethod
        def apply(action: Any) -> None:
            _ = action
            return None

        @staticmethod
        def is_terminal() -> bool:
            return False

        @staticmethod
        def build_result(*, result: str, reason: str | None) -> GameResult:
            return GameResult(
                winner=None,
                result=result,
                reason=reason,
                move_count=3,
                illegal_move_count=0,
                final_board="board",
                move_log=[],
            )

        def finalize_replay(self, result: GameResult) -> GameResult:
            self.finalize_calls += 1
            return GameResult(
                winner=result.winner,
                result=result.result,
                reason=result.reason,
                replay_path="manual_replay.json",
                move_count=result.move_count,
                illegal_move_count=result.illegal_move_count,
                final_board=result.final_board,
                move_log=result.move_log,
                rule_profile=result.rule_profile,
                win_direction=result.win_direction,
                line_length=result.line_length,
                arena_trace=result.arena_trace,
            )

    controller = arena_module._WsRgbSessionController(  # noqa: SLF001
        display_id="display-1",
        sample_id="sample-1",
        adapter_id="arena",
        game_id="retro_env_v1",
    )
    base_env = _BaseEnv()
    wrapped = arena_module._SessionControlledEnvironment(base_env, controller)  # noqa: SLF001

    controller.request_game_end()
    result = wrapped.build_result(result="draw", reason="terminated")

    assert wrapped.is_terminal() is True
    assert result.reason == "manual_stop"
    assert result.replay_path == "manual_replay.json"
    assert base_env.finalize_calls == 1


def test_invoke_sync_waits_for_ws_rgb_process_confirmation(monkeypatch) -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "vizdoom_env_v1", "display_mode": "websocket"},
        players=[{"player_id": "player_0", "type": "human", "ref": "human"}],
    )

    class _Hub:
        def __init__(self) -> None:
            self.registrations: list[Any] = []

        def register_display(self, registration: Any) -> None:
            self.registrations.append(registration)

    class _StubEnvironment:
        @staticmethod
        def reset() -> None:
            return None

        @staticmethod
        def get_last_frame() -> dict[str, Any]:
            return {"frame_id": 1}

        @staticmethod
        def get_active_player() -> str:
            return "player_0"

        @staticmethod
        def observe(player: str) -> ArenaObservation:
            return ArenaObservation(
                board_text="board",
                legal_moves=["0"],
                active_player=player,
            )

        @staticmethod
        def apply(action: Any) -> None:
            _ = action
            return None

        @staticmethod
        def is_terminal() -> bool:
            return True

        @staticmethod
        def build_result(*, result: str, reason: str | None) -> GameResult:
            return GameResult(
                winner=None,
                result=result,
                reason=reason,
                move_count=1,
                illegal_move_count=0,
                final_board="board",
                move_log=[],
            )

    class _StubScheduler:
        @staticmethod
        def run_loop(environment: Any, players: Any) -> GameResult:
            _ = environment, players
            return _make_result(move_log=[{"player": "player_0", "move": "0"}])

    hub = _Hub()
    monkeypatch.setattr(
        adapter,
        "_normalize_player_specs",
        lambda sample: (
            [{"player_id": "player_0", "type": "human", "ref": "human"}],
            ["player_0"],
            {"player_0": "P0"},
            "player_0",
        ),
    )
    monkeypatch.setattr(adapter, "_build_parser", lambda sample: object())
    monkeypatch.setattr(adapter, "_build_scheduler", lambda sample: _StubScheduler())
    monkeypatch.setattr(adapter, "_ensure_visualizer", lambda sample, player_specs: (None, None))
    monkeypatch.setattr(adapter, "_ensure_action_server", lambda player_specs: (None, None))
    monkeypatch.setattr(adapter, "_build_environment", lambda sample, **kwargs: _StubEnvironment())
    monkeypatch.setattr(adapter, "_build_players", lambda *args, **kwargs: [object()])
    adapter._ensure_ws_rgb_hub = lambda: hub  # type: ignore[method-assign]

    holder: dict[str, Any] = {}

    def _run() -> None:
        holder["output"] = adapter._invoke_sync(
            {"sample": {"id": "sample-wait", "task_id": "task-wait", "metadata": {}}, "role_manager": object()},
            RoleAdapterState(),
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    deadline = time.time() + 2.0
    while not hub.registrations and time.time() < deadline:
        time.sleep(0.01)
    assert hub.registrations

    registration = hub.registrations[0]
    assert callable(registration.session_state_source)
    while registration.session_state_source()["phase"] != "game_ended":
        assert time.time() < deadline
        time.sleep(0.01)

    assert thread.is_alive() is True
    response = registration.terminate_process(confirm=True)
    thread.join(timeout=1.0)

    assert response["ok"] is True
    assert response["session"]["phase"] == "process_ended"
    assert thread.is_alive() is False
    assert holder["output"]["result"] == "draw"


def test_format_result_keeps_small_game_log_and_trace_fields() -> None:
    adapter = ArenaRoleAdapter(adapter_id="arena")
    result = _make_result(
        move_log=[{"turn": 1}],
        replay_path="replay/sample.json",
        arena_trace=[{"step_index": 1}],
    )

    output = adapter._format_result(result, {"id": "sample-1"}, None)

    assert output["game_log"] == [{"turn": 1}]
    assert output["replay_path"] == "replay/sample.json"
    assert output["arena_trace"] == [{"step_index": 1}]


def test_ensure_visualizer_skips_string_false_enabled(monkeypatch) -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        visualizer={"enabled": "false"},
    )

    def _fail_constructor(**kwargs: Any) -> None:
        raise AssertionError(f"visualizer should not be created: {kwargs}")

    monkeypatch.setattr(gradio_visualizer_module, "GradioVisualizer", _fail_constructor)

    visualizer, action_queue = adapter._ensure_visualizer(
        {"metadata": {}},
        [{"player_id": "p0", "type": "human"}],
    )

    assert visualizer is None
    assert action_queue is None


def test_ensure_visualizer_coerces_string_bool_flags(monkeypatch) -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "doudizhu_arena_v1"},
        visualizer={
            "enabled": "true",
            "launch_browser": "false",
            "wait_for_finish": "off",
            "sanitize_output": "0",
            "max_output_entries": "7",
            "show_parsed_move": "no",
            "show_chat": "false",
            "allow_status_html": "0",
            "demo_mode": "off",
            "auto_close": "false",
        },
    )
    captured: dict[str, Any] = {}

    class _StubVisualizer:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.action_queue = object()

        def start(self) -> None:
            captured["started"] = True

    monkeypatch.setattr(gradio_visualizer_module, "GradioVisualizer", _StubVisualizer)

    visualizer, action_queue = adapter._ensure_visualizer(
        {"metadata": {}},
        [{"player_id": "p0", "type": "human"}],
    )

    assert visualizer is not None
    assert action_queue is visualizer.action_queue
    assert captured["started"] is True
    assert captured["launch_browser"] is False
    assert captured["wait_for_finish"] is False
    assert captured["sanitize_output"] is False
    assert captured["max_output_entries"] == 7
    assert captured["renderer_impl"] == "doudizhu_showdown_v1"
    assert captured["show_parsed_move"] is False
    assert captured["show_chat"] is False
    assert captured["allow_status_html"] is False
    assert captured["demo_mode"] is False
    assert captured["auto_close"] is False


def test_ensure_action_server_skips_string_false_enabled(monkeypatch) -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        human_input={"enabled": "false"},
    )

    def _fail_constructor(**kwargs: Any) -> None:
        raise AssertionError(f"action server should not be created: {kwargs}")

    monkeypatch.setattr(action_server_module, "ActionQueueServer", _fail_constructor)

    server, action_queue = adapter._ensure_action_server(
        [{"player_id": "p0", "type": "human"}]
    )

    assert server is None
    assert action_queue is None


def test_format_game_log_returns_preview_when_no_run_dir(monkeypatch) -> None:
    adapter = ArenaRoleAdapter(adapter_id="arena")
    monkeypatch.delenv("GAGE_EVAL_RUN_ID", raising=False)
    monkeypatch.setenv("GAGE_EVAL_GAME_LOG_INLINE_LIMIT", "0")
    monkeypatch.setenv("GAGE_EVAL_GAME_LOG_PREVIEW_LIMIT", "1")

    output = adapter._format_game_log(
        [{"idx": 1}, {"idx": 2}], {"id": "sample-2"}, trace=None
    )

    assert "game_log_path" not in output
    assert output["game_log_total"] == 2
    assert output["game_log_truncated"] is True
    assert len(output["game_log_preview"]) == 1


def test_game_log_helper_limits_and_env_parsing(monkeypatch) -> None:
    adapter = ArenaRoleAdapter(adapter_id="arena")
    move_log = [{"move": "very-long-move"}]

    monkeypatch.setenv("GAGE_EVAL_GAME_LOG_INLINE_LIMIT", "0")
    assert adapter._should_externalize_game_log(move_log) is True

    monkeypatch.setenv("GAGE_EVAL_GAME_LOG_INLINE_LIMIT", "-1")
    monkeypatch.setenv("GAGE_EVAL_GAME_LOG_INLINE_BYTES", "0")
    assert adapter._should_externalize_game_log(move_log) is False

    monkeypatch.setenv("GAGE_EVAL_GAME_LOG_INLINE_BYTES", "5")
    assert adapter._should_externalize_game_log(move_log) is True

    monkeypatch.setattr(adapter, "_estimate_game_log_bytes", lambda _move_log: None)
    assert adapter._should_externalize_game_log(move_log) is True

    monkeypatch.setenv("GAGE_EVAL_GAME_LOG_PREVIEW_LIMIT", "1")
    preview = adapter._preview_game_log([{"idx": 1}, {"idx": 2}])
    assert preview["game_log_total"] == 2
    assert preview["game_log_truncated"] is True
    assert len(preview["game_log_preview"]) == 1

    monkeypatch.setenv("GAGE_EVAL_GAME_LOG_INLINE_BYTES", "invalid")
    assert adapter._read_int_env("GAGE_EVAL_GAME_LOG_INLINE_BYTES", 42) == 42


def test_write_game_log_returns_none_when_directory_creation_fails(tmp_path) -> None:
    adapter = ArenaRoleAdapter(adapter_id="arena")
    run_dir = tmp_path / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").write_text("blocked", encoding="utf-8")

    output_path = adapter._write_game_log(run_dir, "sample/1", {"move_log": []})

    assert output_path is None


def test_resolve_run_dir_and_sample_id_fallbacks(tmp_path, monkeypatch) -> None:
    adapter = ArenaRoleAdapter(adapter_id="arena")
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    monkeypatch.setenv("GAGE_EVAL_RUN_ID", "run-from-env")

    run_dir = adapter._resolve_run_dir(trace=None)

    assert run_dir == (tmp_path / "run-from-env").resolve()
    assert adapter._resolve_sample_id({"sample_id": 123}) == "123"
    assert adapter._resolve_sample_id({}) == "sample"
    assert adapter._sanitize_filename(" sample/id:? ") == "sample_id"


def test_estimate_game_log_bytes_returns_none_on_json_error(monkeypatch) -> None:
    def _raise_type_error(*args: Any, **kwargs: Any) -> str:
        raise TypeError("bad json")

    monkeypatch.setattr(arena_module.json, "dumps", _raise_type_error)

    assert ArenaRoleAdapter._estimate_game_log_bytes([{"idx": 1}]) is None


def test_visualized_environment_uses_last_action_in_status() -> None:
    class _BaseEnv:
        def reset(self) -> None:
            return None

        def get_active_player(self) -> str:
            return "p0"

        def observe(self, player: str) -> ArenaObservation:
            return ArenaObservation(
                board_text="board-state",
                legal_moves=["A1"],
                active_player=player,
                last_move="A1",
                metadata={"player_names": {"p0": "Alice"}},
            )

    class _Visualizer:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def update(self, **kwargs: Any) -> None:
            self.calls.append(kwargs)

    visualizer = _Visualizer()
    wrapped = _VisualizedEnvironment(_BaseEnv(), visualizer)

    wrapped.reset()

    assert visualizer.calls
    latest = visualizer.calls[-1]
    assert latest["status_text"] == "Turn: Alice Last: A1"
    assert latest["board_text"] == "board-state"
    assert latest["last_move"] == "A1"


def test_invoke_sync_collects_frame_events_when_frame_capture_enabled(
    tmp_path, monkeypatch
) -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={
            "impl": "gomoku_local_v1",
            "replay": {
                "enabled": True,
                "frame_capture": {
                    "enabled": True,
                    "frame_stride": 1,
                    "max_frames": 0,
                },
            },
        },
        players=[{"player_id": "player_0", "type": "backend", "ref": "dummy_backend"}],
    )
    trace = ObservabilityTrace(run_id="run_frame_capture")
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))

    class _StubEnvironment:
        def __init__(self) -> None:
            self._frame_counter = 0

        @staticmethod
        def reset() -> None:
            return None

        def get_last_frame(self) -> dict[str, Any]:
            self._frame_counter += 1
            return {
                "board_text": f"frame-{self._frame_counter}",
                "move_count": self._frame_counter,
            }

        @staticmethod
        def get_active_player() -> str:
            return "player_0"

        @staticmethod
        def observe(player: str) -> ArenaObservation:
            return ArenaObservation(
                board_text="board",
                legal_moves=["A1"],
                active_player=player,
                last_move=None,
            )

        @staticmethod
        def apply(action: Any) -> None:
            return None

        @staticmethod
        def is_terminal() -> bool:
            return True

    class _StubScheduler:
        @staticmethod
        def run_loop(environment: Any, players: Any) -> GameResult:
            environment.reset()
            return _make_result(move_log=[])

    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        adapter,
        "_normalize_player_specs",
        lambda sample: (
            [{"player_id": "player_0", "type": "backend", "ref": "dummy_backend"}],
            ["player_0"],
            {"player_0": "player_0"},
            "player_0",
        ),
    )
    monkeypatch.setattr(adapter, "_build_parser", lambda sample: object())
    monkeypatch.setattr(adapter, "_build_scheduler", lambda sample: _StubScheduler())
    monkeypatch.setattr(
        adapter, "_ensure_visualizer", lambda sample, player_specs: (None, None)
    )
    monkeypatch.setattr(
        adapter, "_ensure_action_server", lambda player_specs: (None, None)
    )
    monkeypatch.setattr(
        adapter, "_build_environment", lambda sample, **kwargs: _StubEnvironment()
    )
    monkeypatch.setattr(adapter, "_build_players", lambda *args, **kwargs: [object()])

    def _capture_replay_write(
        *,
        result: GameResult,
        sample: dict[str, Any],
        trace: ObservabilityTrace | None,
        output: dict[str, Any],
        frame_events: list[dict[str, Any]] | None = None,
    ) -> None:
        captured["frame_events"] = list(frame_events or [])
        return None

    monkeypatch.setattr(adapter, "_maybe_write_replay_v1", _capture_replay_write)

    adapter._invoke_sync(
        {
            "sample": {"id": "sample-frame", "metadata": {}},
            "role_manager": object(),
            "trace": trace,
        },
        RoleAdapterState(),
    )

    frame_events = captured.get("frame_events")
    assert isinstance(frame_events, list)
    assert len(frame_events) >= 1
    assert frame_events[0]["type"] == "frame"


def test_frame_capture_environment_skips_empty_reset_snapshot() -> None:
    captured: list[dict[str, Any]] = []

    class _Recorder:
        def capture(
            self,
            frame_payload: Any,
            *,
            step: int,
            actor: str | None = None,
            force: bool = False,
        ) -> None:
            captured.append(
                {
                    "frame_payload": frame_payload,
                    "step": step,
                    "actor": actor,
                    "force": force,
                }
            )

    class _Action:
        def __init__(self, player: str) -> None:
            self.player = player

    class _BaseEnv:
        def __init__(self) -> None:
            self._frames = [{}, {"board_text": "ready"}]
            self._index = 0

        def reset(self) -> None:
            return None

        def get_last_frame(self) -> dict[str, Any]:
            value = self._frames[min(self._index, len(self._frames) - 1)]
            self._index += 1
            return dict(value)

        @staticmethod
        def apply(action: Any) -> None:
            _ = action
            return None

    wrapped = arena_module._FrameCaptureEnvironment(_BaseEnv(), _Recorder())  # noqa: SLF001
    wrapped.reset()

    assert captured == []

    wrapped.apply(_Action("player_0"))
    assert len(captured) == 1
    assert captured[0]["step"] == 1
    assert captured[0]["actor"] == "player_0"
    assert captured[0]["frame_payload"]["board_text"] == "ready"
