from types import SimpleNamespace

import pytest

from gage_eval.game_kits.aec_env_game.pettingzoo.environment import (
    PettingZooAecArenaEnvironment,
)
from gage_eval.game_kits.aec_env_game.pettingzoo.envs import space_invaders as module


def _player_specs():
    return (
        SimpleNamespace(player_id="pilot_alpha", display_name="Pilot Alpha"),
        SimpleNamespace(player_id="pilot_beta", display_name="Pilot Beta"),
    )


def test_space_invaders_uses_gamekit_owned_pettingzoo_environment():
    assert module.PettingZooAecArenaEnvironment is PettingZooAecArenaEnvironment


def test_space_invaders_prefers_real_env_with_rgb_defaults(monkeypatch):
    calls: list[dict[str, object]] = []

    def _fake_adapter(**kwargs):
        calls.append(dict(kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(module, "PettingZooAecArenaEnvironment", _fake_adapter)

    environment = module.SpaceInvadersEnvironment(
        env_id="pettingzoo.atari.space_invaders_v2",
        max_cycles=7,
        seed=13,
        use_action_meanings=True,
        include_raw_obs=False,
        illegal_policy={"retry": 0, "on_fail": "loss"},
        action_labels=None,
        player_specs=_player_specs(),
        backend_mode="auto",
        env_kwargs={"difficulty": 2},
    )

    assert environment is not None
    assert len(calls) == 1
    assert "env" not in calls[0]
    assert calls[0]["env_id"] == "pettingzoo.atari.space_invaders_v2"
    assert calls[0]["env_kwargs"] == {
        "difficulty": 2,
        "render_mode": "rgb_array",
        "max_cycles": 7,
    }


def test_space_invaders_auto_mode_falls_back_to_stub_when_real_env_fails(monkeypatch):
    calls: list[dict[str, object]] = []

    def _fake_adapter(**kwargs):
        calls.append(dict(kwargs))
        if "env" not in kwargs:
            raise RuntimeError("real backend unavailable")
        return SimpleNamespace()

    monkeypatch.setattr(module, "PettingZooAecArenaEnvironment", _fake_adapter)

    environment = module.SpaceInvadersEnvironment(
        env_id="pettingzoo.atari.space_invaders_v2",
        max_cycles=5,
        seed=7,
        use_action_meanings=True,
        include_raw_obs=True,
        illegal_policy=None,
        action_labels=("NOOP", "FIRE"),
        player_specs=_player_specs(),
        backend_mode="auto",
    )

    assert environment is not None
    assert len(calls) == 2
    assert "env" not in calls[0]
    assert isinstance(calls[1]["env"], module._StubSpaceInvadersAecEnv)


def test_space_invaders_real_mode_raises_when_real_env_fails(monkeypatch):
    def _fake_adapter(**kwargs):
        if "env" not in kwargs:
            raise RuntimeError("real backend unavailable")
        return SimpleNamespace()

    monkeypatch.setattr(module, "PettingZooAecArenaEnvironment", _fake_adapter)

    with pytest.raises(RuntimeError, match="real backend unavailable"):
        module.SpaceInvadersEnvironment(
            env_id="pettingzoo.atari.space_invaders_v2",
            max_cycles=5,
            seed=7,
            use_action_meanings=True,
            include_raw_obs=True,
            illegal_policy=None,
            action_labels=None,
            player_specs=_player_specs(),
            backend_mode="real",
        )
