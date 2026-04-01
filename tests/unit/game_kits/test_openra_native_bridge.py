from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from gage_eval.game_kits.real_time_game.openra.native_bridge import (
    OPENRA_BRIDGE_DIR_ENV,
    OpenRANativeBridge,
    _build_launch_arguments,
    _encode_startup_orders,
)

_SMALL_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAGUlEQVR4nGNkaGBgYGBg+M8ABYwMjAyMDAwAAB0vAQx0J7s8AAAAAElFTkSuQmCC"
)


def test_openra_native_bridge_launch_args_force_windowed_resolution() -> None:
    args = _build_launch_arguments(
        launch_script=Path("/tmp/launch-game.sh"),
        mod_id="ra",
        support_dir=Path("/tmp/openra-support"),
        viewport={"width": 1280, "height": 720},
        map_ref="map/openra/ra/marigold-town.oramap",
        launch_mode=None,
    )

    assert args[:2] == [
        "/tmp/launch-game.sh",
        "Game.Mod=ra",
    ]
    assert "Engine.SupportDir=/tmp/openra-support" in args
    assert "Graphics.Mode=Windowed" in args
    assert "Graphics.WindowedSize=1280,720" in args
    assert "Graphics.VSync=False" in args
    assert "Graphics.UIScale=1" in args
    assert "Launch.Map=marigold-town.oramap" in args


def test_openra_native_bridge_launch_args_include_skirmish_mode_when_requested() -> None:
    args = _build_launch_arguments(
        launch_script=Path("/tmp/launch-game.sh"),
        mod_id="ra",
        support_dir=Path("/tmp/openra-support"),
        viewport={"width": 1280, "height": 720},
        map_ref="map/openra/ra/marigold-town.oramap",
        launch_mode="skirmish",
    )

    assert "Launch.Map=marigold-town.oramap" in args
    assert "Launch.Mode=skirmish" in args


def test_openra_native_bridge_encodes_only_non_blank_startup_orders() -> None:
    encoded = _encode_startup_orders(
        [
            " option fog False ",
            "",
            "   ",
            "option explored True",
        ]
    )

    assert json.loads(encoded or "[]") == [
        "option fog False",
        "option explored True",
    ]


def test_openra_native_bridge_load_state_skips_unready_frames(tmp_path: Path) -> None:
    bridge = OpenRANativeBridge(
        env_id="ra_skirmish_1v1",
        mod_id="ra",
        map_ref="map/openra/ra/marigold-town.oramap",
        reference_root=tmp_path,
        support_dir=tmp_path / "support",
        bridge_dir=tmp_path / "bridge",
    )
    bridge._frame_path.write_bytes(_SMALL_PNG_BYTES)
    bridge._state_path.write_text(
        json.dumps(
            {
                "mime_type": "image/png",
                "engine_tick": 3,
                "viewport": {"width": 1280, "height": 720},
                "is_ready": False,
            }
        ),
        encoding="utf-8",
    )

    assert bridge._load_state() is None


def test_openra_native_bridge_load_state_preserves_readiness_and_world_metadata(
    tmp_path: Path,
) -> None:
    bridge = OpenRANativeBridge(
        env_id="ra_skirmish_1v1",
        mod_id="ra",
        map_ref="map/openra/ra/marigold-town.oramap",
        reference_root=tmp_path,
        support_dir=tmp_path / "support",
        bridge_dir=tmp_path / "bridge",
    )
    bridge._frame_path.write_bytes(_SMALL_PNG_BYTES)
    bridge._state_path.write_text(
        json.dumps(
            {
                "mime_type": "image/png",
                "engine_tick": 33,
                "render_frame": 105,
                "updated_at_ms": 123456,
                "is_ready": True,
                "ready_reason": "steady_state",
                "playable_player_count": 2,
                "local_player_actor_count": 2,
                "camera_center": {"x": 64000, "y": 15872, "z": 0},
                "viewport": {"width": 1280, "height": 720},
            }
        ),
        encoding="utf-8",
    )

    loaded = bridge._load_state()

    assert loaded is not None
    assert loaded["is_ready"] is True
    assert loaded["ready_reason"] == "steady_state"
    assert loaded["playable_player_count"] == 2
    assert loaded["local_player_actor_count"] == 2
    assert loaded["camera_center"] == {"x": 64000, "y": 15872, "z": 0}


def test_openra_native_bridge_prefers_bridge_dir_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override_dir = tmp_path / "bridge-override"
    monkeypatch.setenv(OPENRA_BRIDGE_DIR_ENV, str(override_dir))

    bridge = OpenRANativeBridge(
        env_id="ra_skirmish_1v1",
        mod_id="ra",
        map_ref="map/openra/ra/marigold-town.oramap",
        reference_root=tmp_path,
        support_dir=tmp_path / "support",
    )

    assert bridge._bridge_dir == override_dir.resolve()
