from __future__ import annotations

import pytest

from gage_eval.role.arena.games.retro.action_codec import RetroActionCodec
from gage_eval.role.arena.games.retro.observation import InfoDeltaFeeder, InfoLastFeeder
from gage_eval.role.arena.games.retro.retro_env import StableRetroArenaEnvironment
from gage_eval.role.arena.types import ArenaAction


class _DummyRetro:
    def __init__(self, *, step_result, buttons=None) -> None:
        self._step_result = step_result
        self.buttons = buttons or ["LEFT", "RIGHT", "A"]
        self.render_calls = 0
        self.reset_calls = 0

    def reset(self, seed=None):
        _ = seed
        self.reset_calls += 1
        return ("obs0", {"x": 0})

    def step(self, buttons):
        _ = buttons
        return self._step_result

    def render(self):
        self.render_calls += 1


def test_retro_env_reset_and_observe(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _DummyRetro(step_result=("obs1", 0.0, False, False, {"x": 1}))
    env = StableRetroArenaEnvironment(game="TestGame", render=True, render_every_n_ticks=1)

    monkeypatch.setattr(env, "_make_env", lambda: dummy)
    env.reset()

    obs = env.observe("player_0")
    assert obs.board_text
    assert obs.metadata.get("contract_v15")
    assert dummy.reset_calls == 1


def test_retro_env_apply_illegal_move_terminates(monkeypatch: pytest.MonkeyPatch) -> None:
    step_result = ("obs1", 1.0, True, False, {"win": True})
    dummy = _DummyRetro(step_result=step_result)
    env = StableRetroArenaEnvironment(game="TestGame")

    monkeypatch.setattr(env, "_make_env", lambda: dummy)
    env.reset()

    action = ArenaAction(player="player_0", move="invalid", raw="invalid", metadata={"hold_ticks": 1})
    result = env.apply(action)

    assert result is not None
    assert result.result == "win"
    assert env.is_terminal()
    assert env._illegal_move_count == 1


def test_retro_env_normalize_step_variants() -> None:
    obs, reward, terminated, truncated, info = StableRetroArenaEnvironment._normalize_step(
        ("obs", 1.0, True, False, {"x": 1})
    )
    assert terminated is True
    assert truncated is False
    assert info["x"] == 1

    obs, reward, terminated, truncated, info = StableRetroArenaEnvironment._normalize_step(
        ("obs", 0.0, True, {"x": 2})
    )
    assert terminated is True
    assert truncated is False
    assert info["x"] == 2

    with pytest.raises(ValueError):
        StableRetroArenaEnvironment._normalize_step(("obs", 0.0))


def test_retro_env_normalize_reset_variants() -> None:
    obs, info = StableRetroArenaEnvironment._normalize_reset(("obs", {"x": 1}))
    assert info["x"] == 1
    obs, info = StableRetroArenaEnvironment._normalize_reset("obs")
    assert info == {}


def test_retro_env_maybe_render_respects_stride(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _DummyRetro(step_result=("obs1", 0.0, False, False, {}))
    env = StableRetroArenaEnvironment(game="TestGame", render=True, render_every_n_ticks=2)
    monkeypatch.setattr(env, "_make_env", lambda: dummy)
    env.reset()
    dummy.render_calls = 0

    env._tick = 1
    env._maybe_render()
    assert dummy.render_calls == 0

    env._maybe_render(force=True)
    assert dummy.render_calls == 1


def test_retro_env_build_helpers() -> None:
    env = StableRetroArenaEnvironment(game="TestGame")
    feeder = env._build_info_feeder({"impl": "info_delta_v1", "params": {"window_size": 3}})
    assert isinstance(feeder, InfoDeltaFeeder)
    feeder = env._build_info_feeder({"impl": "info_last_v1"})
    assert isinstance(feeder, InfoLastFeeder)

    schema = env._build_action_schema({"hold_ticks_min": 2, "hold_ticks_max": 5, "hold_ticks_default": 3})
    assert schema.hold_ticks_min == 2
    assert schema.hold_ticks_max == 5
    assert schema.default_hold_ticks == 3


def test_retro_env_build_action_codec_uses_buttons() -> None:
    env = StableRetroArenaEnvironment(game="TestGame")
    dummy = _DummyRetro(step_result=("obs1", 0.0, False, False, {}), buttons=["LEFT", "RIGHT", "A"])
    codec = env._build_action_codec(dummy)
    assert isinstance(codec, RetroActionCodec)
    assert "noop" in codec.legal_moves()
