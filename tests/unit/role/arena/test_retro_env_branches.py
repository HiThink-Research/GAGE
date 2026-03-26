import builtins
import sys
import types
from pathlib import Path

import pytest

from gage_eval.role.arena.games.retro import retro_env as retro_env_mod
from gage_eval.role.arena.types import GameResult


def _result() -> GameResult:
    return GameResult(
        winner=None,
        result="draw",
        reason=None,
        move_count=0,
        illegal_move_count=0,
        final_board="{}",
        move_log=[],
    )


def test_finalize_replay_returns_original_result_when_writer_missing_or_empty_path():
    env = retro_env_mod.StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="headless")
    result = _result()

    assert env.finalize_replay(result) is result

    class StubWriter:
        def finalize(self, _):
            return None

    env._replay_writer = StubWriter()  # type: ignore[attr-defined]
    assert env.finalize_replay(result) is result


def test_ensure_env_builds_runtime_and_refreshes_when_runtime_policy_is_fresh(monkeypatch: pytest.MonkeyPatch):
    env = retro_env_mod.StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="headless")
    sentinel_env = object()
    sentinel_codec = object()
    calls = {"make": 0, "codec": 0}

    def fake_make():
        calls["make"] += 1
        return sentinel_env

    def fake_codec(raw_env):
        calls["codec"] += 1
        assert raw_env is sentinel_env
        return sentinel_codec

    monkeypatch.setattr(env, "_make_env", fake_make)
    monkeypatch.setattr(env, "_build_action_codec", fake_codec)

    env._ensure_env()  # noqa: SLF001
    assert env._retro_env is sentinel_env  # noqa: SLF001
    assert env._action_codec is sentinel_codec  # noqa: SLF001

    env._ensure_env()  # noqa: SLF001
    assert calls["make"] == 1

    env._runtime_policy = "fresh"  # noqa: SLF001
    env._ensure_env()  # noqa: SLF001
    assert calls["make"] == 2
    assert calls["codec"] == 2


def test_reset_state_raises_when_runtime_not_initialized():
    env = retro_env_mod.StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="headless")
    with pytest.raises(RuntimeError, match="not initialized"):
        env._reset_state()  # noqa: SLF001


def test_make_env_ignores_record_target_when_parent_mkdir_fails(monkeypatch: pytest.MonkeyPatch, tmp_path):
    calls: list[dict[str, object]] = []

    class StubRetro:
        @staticmethod
        def make(**kwargs):
            calls.append(dict(kwargs))
            return {"env": "ok"}

    env = retro_env_mod.StableRetroArenaEnvironment(
        game="SuperMarioBros3-Nes-v0",
        display_mode="headless",
        record_path=str(tmp_path / "nested" / "movie.bk2"),
    )
    monkeypatch.setattr(retro_env_mod.importlib, "import_module", lambda _: StubRetro)
    monkeypatch.setattr(Path, "mkdir", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("boom")))

    assert env._make_env() == {"env": "ok"}  # noqa: SLF001
    assert calls[0] == {"game": "SuperMarioBros3-Nes-v0", "state": "Start", "render_mode": "rgb_array"}


def test_make_env_uses_render_mode_fallback_without_record(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []

    class StubRetro:
        @staticmethod
        def make(**kwargs):
            calls.append(dict(kwargs))
            if "state" in kwargs:
                raise TypeError("state unsupported")
            return {"env": "ok"}

    env = retro_env_mod.StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="headless")
    monkeypatch.setattr(retro_env_mod.importlib, "import_module", lambda _: StubRetro)

    assert env._make_env() == {"env": "ok"}  # noqa: SLF001
    assert calls[1] == {"game": "SuperMarioBros3-Nes-v0", "render_mode": "rgb_array"}


def test_maybe_render_handles_missing_env_and_show_frame_errors(monkeypatch: pytest.MonkeyPatch):
    env = retro_env_mod.StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="pil")

    env._retro_env = None  # type: ignore[attr-defined]
    env._maybe_render(force=True)  # noqa: SLF001

    env._retro_env = object()  # type: ignore[attr-defined]
    env._tick = 1  # type: ignore[attr-defined]
    env._last_obs = None  # type: ignore[attr-defined]
    env._maybe_render(force=True)  # noqa: SLF001

    env._last_obs = object()  # type: ignore[attr-defined]
    monkeypatch.setattr(env, "_show_frame", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    env._maybe_render(force=True)  # noqa: SLF001


def test_resolve_record_output_path_branches(tmp_path):
    with_record_path = retro_env_mod.StableRetroArenaEnvironment(
        game="SuperMarioBros3-Nes-v0",
        display_mode="headless",
        record_path=str(tmp_path / "explicit.bk2"),
    )
    assert with_record_path._resolve_record_output_path() == Path(tmp_path / "explicit.bk2")  # noqa: SLF001

    no_record = retro_env_mod.StableRetroArenaEnvironment(
        game="SuperMarioBros3-Nes-v0",
        display_mode="headless",
        record_bk2=False,
    )
    assert no_record._resolve_record_output_path() is None  # noqa: SLF001

    unresolved = retro_env_mod.StableRetroArenaEnvironment(
        game="SuperMarioBros3-Nes-v0",
        display_mode="headless",
        record_bk2=True,
    )
    assert unresolved._resolve_record_output_path() is None  # noqa: SLF001


def test_build_action_codec_raises_when_buttons_are_missing():
    class DummyEnv:
        buttons = []

    env = retro_env_mod.StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="headless")
    with pytest.raises(ValueError, match="missing buttons"):
        env._build_action_codec(DummyEnv())  # noqa: SLF001


def test_show_frame_covers_non_rgb_viewer_disabled_and_viewer_errors(monkeypatch: pytest.MonkeyPatch):
    np = pytest.importorskip("numpy")
    env = retro_env_mod.StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="headless")

    env._show_frame(None)  # noqa: SLF001
    env._show_frame(np.zeros((2, 2), dtype=np.uint8))  # noqa: SLF001

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    env._tick = 1  # type: ignore[attr-defined]
    env._frame_viewer_disabled = True  # type: ignore[attr-defined]
    env._show_frame(frame)  # noqa: SLF001

    env._frame_viewer_disabled = False  # type: ignore[attr-defined]
    env._frame_viewer = None  # type: ignore[attr-defined]
    monkeypatch.setattr(env, "_init_frame_viewer", lambda: None)
    env._show_frame(frame)  # noqa: SLF001
    assert env._frame_viewer_disabled is True  # noqa: SLF001

    class DummyRoot:
        def update_idletasks(self):
            return None

        def update(self):
            return None

    class DummyLabel:
        def __init__(self) -> None:
            self.last_image = None

        def configure(self, *, image):
            self.last_image = image

    env._frame_viewer_kind = "pil"  # type: ignore[attr-defined]
    env._frame_viewer_disabled = False  # type: ignore[attr-defined]
    env._frame_viewer = {  # type: ignore[attr-defined]
        "root": DummyRoot(),
        "label": DummyLabel(),
        "photo_factory": lambda _arr: "image_ref",
        "image": None,
    }
    env._show_frame(frame)  # noqa: SLF001
    assert env._frame_viewer["image"] == "image_ref"  # noqa: SLF001

    class DummyViewer:
        def imshow(self, _):
            raise RuntimeError("viewer boom")

    env._frame_viewer_kind = "pyglet"  # type: ignore[attr-defined]
    env._frame_viewer = DummyViewer()  # type: ignore[attr-defined]
    env._show_frame(frame)  # noqa: SLF001


def test_init_frame_viewer_handles_import_failures(monkeypatch: pytest.MonkeyPatch):
    env = retro_env_mod.StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="headless")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"tkinter", "stable_retro.rendering"}:
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    env._frame_viewer_kind = "pil"  # type: ignore[attr-defined]
    assert env._init_frame_viewer() is None  # noqa: SLF001

    env._frame_viewer_kind = "pyglet"  # type: ignore[attr-defined]
    assert env._init_frame_viewer() is None  # noqa: SLF001


def test_init_frame_viewer_supports_pil_and_pyglet_paths(monkeypatch: pytest.MonkeyPatch):
    env = retro_env_mod.StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="headless")

    class DummyRoot:
        def title(self, *_):
            return None

    class DummyLabel:
        def __init__(self, _root):
            self.packed = False

        def pack(self):
            self.packed = True

    fake_tk = types.ModuleType("tkinter")
    fake_tk.Tk = DummyRoot
    fake_tk.Label = DummyLabel

    fake_image = types.ModuleType("PIL.Image")
    fake_image.fromarray = lambda arr: arr

    fake_image_tk = types.ModuleType("PIL.ImageTk")

    class FakePhotoImage:
        def __init__(self, image):
            self.image = image

    fake_image_tk.PhotoImage = FakePhotoImage

    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = fake_image
    fake_pil.ImageTk = fake_image_tk

    fake_rendering = types.ModuleType("stable_retro.rendering")

    class FakeSimpleImageViewer:
        pass

    fake_rendering.SimpleImageViewer = FakeSimpleImageViewer
    fake_stable_retro = types.ModuleType("stable_retro")
    fake_stable_retro.rendering = fake_rendering

    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image)
    monkeypatch.setitem(sys.modules, "PIL.ImageTk", fake_image_tk)
    monkeypatch.setitem(sys.modules, "stable_retro", fake_stable_retro)
    monkeypatch.setitem(sys.modules, "stable_retro.rendering", fake_rendering)

    env._frame_viewer_kind = "pil"  # type: ignore[attr-defined]
    pil_viewer = env._init_frame_viewer()  # noqa: SLF001
    assert isinstance(pil_viewer, dict)
    assert callable(pil_viewer["photo_factory"])

    env._frame_viewer_kind = "pyglet"  # type: ignore[attr-defined]
    pyglet_viewer = env._init_frame_viewer()  # noqa: SLF001
    assert isinstance(pyglet_viewer, FakeSimpleImageViewer)


def test_build_info_feeder_unknown_impl_falls_back_to_info_last():
    env = retro_env_mod.StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="headless")
    feeder = env._build_info_feeder({"impl": "custom_unknown", "params": {}})  # noqa: SLF001
    assert isinstance(feeder, retro_env_mod.InfoLastFeeder)


def test_build_info_feeder_none_impl_disables_info_projection():
    env = retro_env_mod.StableRetroArenaEnvironment(game="SuperMarioBros3-Nes-v0", display_mode="headless")
    feeder = env._build_info_feeder({"impl": "info_none_v1", "params": {}})  # noqa: SLF001
    assert isinstance(feeder, retro_env_mod.InfoNoneFeeder)
