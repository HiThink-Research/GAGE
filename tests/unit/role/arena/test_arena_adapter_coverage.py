from __future__ import annotations

import asyncio
from typing import Any

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.adapters import arena as arena_module
from gage_eval.role.adapters.arena import ArenaRoleAdapter, _VisualizedEnvironment
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.arena.schedulers.turn_scheduler import TurnScheduler
from gage_eval.role.arena.types import ArenaObservation, GameResult


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
    monkeypatch.setattr(adapter, "_invoke_sync", lambda payload, state: {"ok": True, "payload": payload})

    result = asyncio.run(adapter.ainvoke({"sample_id": "s1"}, RoleAdapterState()))

    assert result == {"ok": True, "payload": {"sample_id": "s1"}}


def test_build_scheduler_supports_turn_type() -> None:
    adapter = ArenaRoleAdapter(adapter_id="arena", scheduler={"type": " TURN "})

    scheduler = adapter._build_scheduler({"eval_config": {"max_turns": 7}})

    assert isinstance(scheduler, TurnScheduler)


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
    monkeypatch.setattr(adapter, "_resolve_player_labels", lambda specs, role_manager: {"p0": "model-001"})
    monkeypatch.setattr(adapter, "_build_parser", lambda sample: object())
    monkeypatch.setattr(adapter, "_build_scheduler", lambda sample: _StubScheduler())
    monkeypatch.setattr(adapter, "_ensure_visualizer", lambda sample, player_specs: (None, None))
    monkeypatch.setattr(adapter, "_ensure_action_server", lambda player_specs: (None, None))
    monkeypatch.setattr(
        adapter,
        "_build_environment",
        lambda sample, **kwargs: captured_env_kwargs.update(kwargs) or object(),
    )
    monkeypatch.setattr(adapter, "_build_players", lambda *args, **kwargs: [object()])

    output = adapter._invoke_sync({"sample": {"metadata": {}}, "role_manager": object()}, RoleAdapterState())

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
    monkeypatch.setattr(adapter, "_resolve_player_labels", lambda specs, role_manager: {"p0": "model-002"})
    monkeypatch.setattr(adapter, "_build_parser", lambda sample: object())
    monkeypatch.setattr(adapter, "_build_scheduler", lambda sample: _StubScheduler())
    monkeypatch.setattr(adapter, "_ensure_visualizer", lambda sample, player_specs: (None, None))
    monkeypatch.setattr(adapter, "_ensure_action_server", lambda player_specs: (None, None))
    monkeypatch.setattr(
        adapter,
        "_build_environment",
        lambda sample, **kwargs: captured_env_kwargs.update(kwargs) or object(),
    )
    monkeypatch.setattr(adapter, "_build_players", lambda *args, **kwargs: [object()])

    adapter._invoke_sync({"sample": {"metadata": {}}, "role_manager": object()}, RoleAdapterState())

    assert captured_env_kwargs["player_names"]["p0"] == "model-002"


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


def test_format_game_log_returns_preview_when_no_run_dir(monkeypatch) -> None:
    adapter = ArenaRoleAdapter(adapter_id="arena")
    monkeypatch.delenv("GAGE_EVAL_RUN_ID", raising=False)
    monkeypatch.setenv("GAGE_EVAL_GAME_LOG_INLINE_LIMIT", "0")
    monkeypatch.setenv("GAGE_EVAL_GAME_LOG_PREVIEW_LIMIT", "1")

    output = adapter._format_game_log([{"idx": 1}, {"idx": 2}], {"id": "sample-2"}, trace=None)

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
