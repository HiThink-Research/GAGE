import pytest

from gage_eval.role.arena.games.retro import retro_env as retro_env_mod


def test_retro_env_make_env_raises_helpful_error_when_retro_missing(monkeypatch: pytest.MonkeyPatch):
    def _raise_import_error(name: str):
        raise ImportError("missing")

    env = retro_env_mod.StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="headless")
    monkeypatch.setattr(retro_env_mod.importlib, "import_module", _raise_import_error)

    with pytest.raises(ImportError, match="stable-retro is required"):
        env._make_env()  # noqa: SLF001


def test_retro_env_make_env_falls_back_when_state_not_supported(monkeypatch: pytest.MonkeyPatch, tmp_path):
    calls: list[dict[str, object]] = []

    class StubRetro:
        @staticmethod
        def make(**kwargs):
            calls.append(dict(kwargs))
            if "state" in kwargs:
                raise TypeError("state unsupported")
            return {"env": "ok"}

    env = retro_env_mod.StableRetroArenaEnvironment(
        game="SuperMarioBros3-Nes-v0",
        display_mode="headless",
        record_bk2=True,
        record_dir=str(tmp_path),
    )
    monkeypatch.setattr(retro_env_mod.importlib, "import_module", lambda name: StubRetro)

    assert env._make_env() == {"env": "ok"}  # noqa: SLF001
    assert calls[0]["game"] == "SuperMarioBros3-Nes-v0"
    assert calls[0]["state"] == "Start"
    assert "record" in calls[0]
    assert calls[1] == {
        "game": "SuperMarioBros3-Nes-v0",
        "record": calls[0]["record"],
        "render_mode": "rgb_array",
    }


def test_retro_env_make_env_falls_back_to_minimal_make_when_render_mode_unsupported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    calls: list[dict[str, object]] = []

    class StubRetro:
        @staticmethod
        def make(**kwargs):
            calls.append(dict(kwargs))
            if "state" in kwargs:
                raise TypeError("state unsupported")
            if "render_mode" in kwargs:
                raise TypeError("render unsupported")
            return {"env": "ok"}

    env = retro_env_mod.StableRetroArenaEnvironment(
        game="SuperMarioBros3-Nes-v0",
        display_mode="headless",
        record_bk2=True,
        record_dir=str(tmp_path),
    )
    monkeypatch.setattr(retro_env_mod.importlib, "import_module", lambda name: StubRetro)

    assert env._make_env() == {"env": "ok"}  # noqa: SLF001
    assert calls[-1] == {"game": "SuperMarioBros3-Nes-v0"}
