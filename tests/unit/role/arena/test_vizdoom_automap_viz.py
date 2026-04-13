from types import SimpleNamespace

from gage_eval.game_kits.real_time_game.vizdoom import automap_viz


def test_automap_window_falls_back_to_noop_when_cv2_window_init_fails(monkeypatch):
    fake_cv2 = SimpleNamespace(
        WINDOW_NORMAL=0,
        namedWindow=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("headless")),
    )

    monkeypatch.setattr(automap_viz, "cv2", fake_cv2)
    monkeypatch.setattr(automap_viz, "plt", None)

    window = automap_viz.AutomapWindow("ViZDoom POV", bgr=False)

    assert isinstance(window._impl, automap_viz._NoopBackend)
    assert window.update(None) is None
    assert window.close() is None
