import pytest

from gage_eval.role.arena.games.retro.action_codec import RetroActionCodec
from gage_eval.role.arena.games.retro.retro_env import StableRetroArenaEnvironment
from gage_eval.role.arena.games.retro import observation as retro_observation
from gage_eval.role.arena.types import ArenaAction, GameResult


class FakeRetroEnv:
    def __init__(self, *, reset_result, step_results):
        self.reset_result = reset_result
        self.step_results = list(step_results)
        self.reset_seeds = []
        self.step_payloads = []

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return self.reset_result

    def step(self, payload):
        self.step_payloads.append(payload)
        return self.step_results.pop(0)


class FakeRetroEnvNoSeed(FakeRetroEnv):
    def reset(self):  # type: ignore[override]
        self.reset_seeds.append(None)
        return self.reset_result


def _make_env_with_stubbed_runtime(*, retro_env, codec, seed=None):
    env = StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="headless", seed=seed)
    env._retro_env = retro_env  # type: ignore[attr-defined]
    env._action_codec = codec  # type: ignore[attr-defined]
    return env


def test_stable_retro_env_requires_non_empty_game_id():
    with pytest.raises(ValueError, match="non-empty game id"):
        StableRetroArenaEnvironment(game="", display_mode="headless")


def test_retro_env_reset_normalizes_info_and_supports_seed():
    frame = object()
    retro_env = FakeRetroEnv(reset_result=(frame, {"score": 1}), step_results=[])
    codec = RetroActionCodec(buttons=["LEFT"])
    env = _make_env_with_stubbed_runtime(retro_env=retro_env, codec=codec, seed=123)

    env.reset()

    assert retro_env.reset_seeds == [123]
    assert env.get_last_frame() is frame

    obs = env.observe("player_0")
    assert '"score": 1' in (obs.view_text or "")


def test_retro_env_reset_falls_back_when_seed_not_supported():
    retro_env = FakeRetroEnvNoSeed(reset_result=(object(), {}), step_results=[])
    codec = RetroActionCodec(buttons=["LEFT"])
    env = _make_env_with_stubbed_runtime(retro_env=retro_env, codec=codec, seed=999)

    env.reset()

    assert retro_env.reset_seeds == [None]


def test_retro_env_apply_handles_illegal_moves_and_builds_terminal_result():
    frame = object()
    retro_env = FakeRetroEnv(
        reset_result=(frame, {"score": 0}),
        step_results=[(frame, 2.0, True, False, {"win": True})],
    )
    codec = RetroActionCodec(buttons=["LEFT"])
    env = _make_env_with_stubbed_runtime(retro_env=retro_env, codec=codec)
    env.reset()

    action = ArenaAction(player="player_0", move="not_a_move", raw="not_a_move")
    env.record_decision(action, start_tick=0, hold_ticks=2)
    result = env.apply(action)

    assert retro_env.step_payloads == [[0]]
    assert result is not None
    assert result.winner == "player_0"
    assert result.status == "win"
    assert result.reason == "terminated"
    assert result.move_count == 1
    assert result.illegal_move_count == 1
    assert env.is_terminal() is True


def test_retro_env_helpers_normalize_step_and_derive_result():
    obs, reward, terminated, truncated, info = StableRetroArenaEnvironment._normalize_step(  # noqa: SLF001
        ("frame", 1.0, True, {"win": True})
    )
    assert obs == "frame"
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info == {"win": True}

    assert StableRetroArenaEnvironment._derive_result(True, False, {"win": True}) == "win"  # noqa: SLF001
    assert StableRetroArenaEnvironment._derive_result(True, True, {"win": True}) == "draw"  # noqa: SLF001


def test_retro_env_resolve_record_output_path_sanitizes_sample_id(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    env = StableRetroArenaEnvironment(
        game="SuperMarioBros3-Nes-v0",
        display_mode="headless",
        record_bk2=True,
        run_id="run_0",
        sample_id="sample 1!!",
    )

    path = env._resolve_record_output_path()  # noqa: SLF001
    assert path is not None
    assert path.name == "retro_movie_sample_1.bk2"


def test_retro_env_build_action_codec_falls_back_to_unwrapped_buttons():
    class DummyUnwrapped:
        buttons = ["LEFT", "RIGHT"]

    class DummyEnv:
        buttons = []
        unwrapped = DummyUnwrapped()

    env = StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="headless")
    codec = env._build_action_codec(DummyEnv())  # noqa: SLF001
    assert codec.buttons() == ["LEFT", "RIGHT"]


def test_retro_env_finalize_replay_returns_updated_game_result():
    class StubWriter:
        def finalize(self, result: GameResult):
            return "/tmp/replay.json"

    env = StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="headless")
    env._replay_writer = StubWriter()  # type: ignore[attr-defined]

    result = GameResult(
        winner=None,
        result="draw",
        reason=None,
        move_count=0,
        illegal_move_count=0,
        final_board="{}",
        move_log=[],
    )

    updated = env.finalize_replay(result)
    assert updated.replay_path == "/tmp/replay.json"


def test_retro_env_websocket_display_mode_treated_as_headless_and_exposes_active_player():
    env = StableRetroArenaEnvironment(
        game="SuperMarioBros3-Nes-v0",
        display_mode="ws",
        player_ids=["p0", "p1"],
    )

    assert env._display_mode == "headless"  # noqa: SLF001
    assert env.get_active_player() == "p0"


def test_retro_env_apply_errors_when_uninitialized_and_returns_final_when_terminal():
    env = StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="headless")
    with pytest.raises(RuntimeError, match="not initialized"):
        env.apply(ArenaAction(player="player_0", move="noop", raw="noop"))

    final = GameResult(
        winner=None,
        result="draw",
        reason="terminated",
        move_count=0,
        illegal_move_count=0,
        final_board="{}",
        move_log=[],
    )
    env._terminal = True  # type: ignore[attr-defined]
    env._final_result = final  # type: ignore[attr-defined]
    assert env.apply(ArenaAction(player="player_0", move="noop", raw="noop")) is final


def test_retro_env_apply_returns_none_when_not_terminal():
    frame = object()
    retro_env = FakeRetroEnv(
        reset_result=(frame, {}),
        step_results=[(frame, 0.5, False, False, {"tick": 1})],
    )
    codec = RetroActionCodec(buttons=["LEFT"])
    env = _make_env_with_stubbed_runtime(retro_env=retro_env, codec=codec)
    env.reset()

    action = ArenaAction(player="player_0", move="noop", raw="noop")
    env.record_decision(action, start_tick=0, hold_ticks=1)
    assert env.apply(action) is None
    assert env.is_terminal() is False


def test_retro_env_apply_attaches_replay_path_when_writer_returns_one():
    class StubWriter:
        def __init__(self):
            self.ticks = []

        def append_decision(self, *args, **kwargs) -> None:
            return None

        def append_tick(self, *args, **kwargs) -> None:
            self.ticks.append((args, kwargs))

        def finalize(self, result: GameResult):
            return "/tmp/replay.json"

    frame = object()
    retro_env = FakeRetroEnv(
        reset_result=(frame, {}),
        step_results=[(frame, 0.0, True, False, {})],
    )
    codec = RetroActionCodec(buttons=["LEFT"])
    env = _make_env_with_stubbed_runtime(retro_env=retro_env, codec=codec)
    env.reset()
    env._replay_writer = StubWriter()  # type: ignore[attr-defined]

    action = ArenaAction(player="player_0", move="noop", raw="noop")
    result = env.apply(action)
    assert result is not None
    assert result.replay_path == "/tmp/replay.json"


def test_retro_env_controls_payload_skips_unknown_moves():
    env = StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="headless")
    env._action_codec = RetroActionCodec(buttons=["LEFT"])  # type: ignore[attr-defined]

    controls = env._build_controls_payload(["left", "unknown"])  # noqa: SLF001
    assert controls["move_aliases"]["left"]["keys_combo"] == "a"
    assert "unknown" not in controls["move_aliases"]


def test_retro_env_builders_enable_info_delta_and_action_schema_overrides():
    env = StableRetroArenaEnvironment(
        game="SuperMarioBros3-Nes-v0",
        display_mode="headless",
        info_feeder={"impl": "delta", "params": {"window_size": 3}},
        action_schema={"hold_ticks_min": 2, "hold_ticks_max": 4, "hold_ticks_default": 3},
    )

    assert isinstance(env._info_feeder, retro_observation.InfoDeltaFeeder)  # noqa: SLF001
    assert env._action_schema.hold_ticks_min == 2  # noqa: SLF001
    assert env._action_schema.hold_ticks_max == 4  # noqa: SLF001
    assert env._action_schema.default_hold_ticks == 3  # noqa: SLF001


def test_retro_env_helpers_raise_on_invalid_step_and_normalize_reset_without_info():
    with pytest.raises(ValueError, match="unexpected format"):
        StableRetroArenaEnvironment._normalize_step(("frame",))  # noqa: SLF001

    obs, info = StableRetroArenaEnvironment._normalize_reset("frame")  # noqa: SLF001
    assert obs == "frame"
    assert info == {}
