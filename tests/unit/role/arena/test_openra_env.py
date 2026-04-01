from __future__ import annotations

import base64
import json
from types import SimpleNamespace
import zipfile
from pathlib import Path

import gage_eval.game_kits.real_time_game.openra.environment as openra_env_module
from gage_eval.role.arena.types import ArenaAction
from gage_eval.game_kits.real_time_game.openra.environment import OpenRAArenaEnvironment

_SMALL_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAGUlEQVR4nGNkaGBgYGBg+M8ABYwMjAyMDAwAAB0vAQx0J7s8AAAAAElFTkSuQmCC"
)
_ALT_SMALL_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAHElEQVR4nGP8z8DAwMDA8J8BhowMjAyMDAwMAAAcWgENJQ0v3QAAAABJRU5ErkJggg=="
)


def _write_directory_map_preview(
    root: Path,
    *,
    mod_id: str,
    map_name: str,
    title: str,
    map_size: tuple[int, int],
) -> None:
    target_dir = root / "mods" / mod_id / "maps" / map_name
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "map.png").write_bytes(_SMALL_PNG_BYTES)
    (target_dir / "map.yaml").write_text(
        "\n".join(
            (
                "MapFormat: 12",
                "",
                f"RequiresMod: {mod_id}",
                "",
                f"Title: {title}",
                "",
                f"MapSize: {map_size[0]},{map_size[1]}",
                "",
                f"Bounds: 1,1,{map_size[0] - 2},{map_size[1] - 2}",
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_oramap_preview(
    root: Path,
    *,
    mod_id: str,
    archive_name: str,
    title: str,
    map_size: tuple[int, int],
) -> None:
    archive_path = root / "mods" / mod_id / "maps" / archive_name
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr("map.png", _SMALL_PNG_BYTES)
        archive.writestr(
            "map.yaml",
            "\n".join(
                (
                    "MapFormat: 12",
                    "",
                    f"RequiresMod: {mod_id}",
                    "",
                    f"Title: {title}",
                    "",
                    f"MapSize: {map_size[0]},{map_size[1]}",
                    "",
                    f"Bounds: 1,1,{map_size[0] - 2},{map_size[1] - 2}",
                    "",
                )
            ),
        )


class _FakeNativeBridge:
    def __init__(self, *states: dict[str, object]) -> None:
        self._states = list(states)
        self._last_state = self._states[0] if self._states else {}
        self.submitted_inputs: list[dict[str, object]] = []

    def read_state(self) -> dict[str, object]:
        if self._states:
            self._last_state = self._states.pop(0)
        return dict(self._last_state)

    def submit_input(self, payload: dict[str, object]) -> None:
        self.submitted_inputs.append(dict(payload))

    def close(self) -> None:
        return None


def _png_data_url(raw_bytes: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(raw_bytes).decode('ascii')}"


def test_openra_env_frame_image_uses_browser_safe_base64_svg_data_url() -> None:
    env = OpenRAArenaEnvironment(
        env_id="ra_skirmish_1v1",
        mod_id="ra",
        map_ref="map/openra/ra/nonexistent.oramap",
    )

    frame = env.get_last_frame()
    image_data_url = frame["view"]["image"]["data_url"]

    assert image_data_url.startswith("data:image/svg+xml;base64,")
    encoded_svg = image_data_url.removeprefix("data:image/svg+xml;base64,")
    decoded_svg = base64.b64decode(encoded_svg).decode("utf-8")

    assert "<svg" in decoded_svg
    assert "url(#bg)" in decoded_svg
    assert "Objective:" in decoded_svg
    assert "Hold the ridge" in decoded_svg


def test_openra_env_prefers_reference_oramap_preview_when_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_oramap_preview(
        tmp_path,
        mod_id="ra",
        archive_name="marigold-town.oramap",
        title="Marigold Town",
        map_size=(99, 99),
    )
    monkeypatch.setenv("GAGE_OPENRA_REFERENCE_ROOT", str(tmp_path))

    env = OpenRAArenaEnvironment(
        env_id="ra_skirmish_1v1",
        mod_id="ra",
        map_ref="map/openra/ra/marigold-town.oramap",
    )

    frame = env.get_last_frame()
    image_payload = frame["view"]["image"]
    map_payload = frame["metadata"]["map"]

    assert image_payload["mimeType"] == "image/png"
    assert image_payload["data_url"].startswith("data:image/png;base64,")
    assert map_payload == {
        "id": "ra_skirmish_1v1",
        "mod_id": "ra",
        "title": "Marigold Town",
        "map_size": {"width": 99, "height": 99},
        "bounds": {"x": 1, "y": 1, "width": 97, "height": 97},
        "image_size": {"width": 4, "height": 4},
        "preview_source": "reference_map_preview",
    }


def test_openra_env_prefers_reference_directory_preview_when_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_directory_map_preview(
        tmp_path,
        mod_id="cnc",
        map_name="gdi01",
        title="Storm the Beachhead",
        map_size=(64, 64),
    )
    monkeypatch.setenv("GAGE_OPENRA_REFERENCE_ROOT", str(tmp_path))

    env = OpenRAArenaEnvironment(
        env_id="cnc_mission_gdi01",
        mod_id="cnc",
        map_ref="map/openra/cnc/gdi01",
    )

    frame = env.get_last_frame()
    image_payload = frame["view"]["image"]
    map_payload = frame["metadata"]["map"]

    assert image_payload["mimeType"] == "image/png"
    assert image_payload["data_url"].startswith("data:image/png;base64,")
    assert map_payload == {
        "id": "cnc_mission_gdi01",
        "mod_id": "cnc",
        "title": "Storm the Beachhead",
        "map_size": {"width": 64, "height": 64},
        "bounds": {"x": 1, "y": 1, "width": 62, "height": 62},
        "image_size": {"width": 4, "height": 4},
        "preview_source": "reference_map_preview",
    }


def test_openra_native_env_uses_bridge_frames_and_routes_browser_input() -> None:
    bridge = _FakeNativeBridge(
        {
            "frame_data_url": _png_data_url(_SMALL_PNG_BYTES),
            "mime_type": "image/png",
            "engine_tick": 16,
            "viewport": {"width": 1280, "height": 720},
        },
        {
            "frame_data_url": _png_data_url(_SMALL_PNG_BYTES),
            "mime_type": "image/png",
            "engine_tick": 17,
            "viewport": {"width": 1280, "height": 720},
        },
        {
            "frame_data_url": _png_data_url(_ALT_SMALL_PNG_BYTES),
            "mime_type": "image/png",
            "engine_tick": 18,
            "viewport": {"width": 1280, "height": 720},
        },
    )
    env = OpenRAArenaEnvironment(
        env_id="cnc_mission_gdi01",
        backend_mode="native",
        mod_id="cnc",
        map_ref="map/openra/cnc/gdi01",
        native_bridge=bridge,
    )

    initial_frame = env.get_last_frame()
    assert initial_frame["view"]["image"] == {
        "data_url": _png_data_url(_SMALL_PNG_BYTES),
        "mimeType": "image/png",
    }
    assert initial_frame["tick"] == 17
    assert initial_frame["metadata"]["map"]["preview_source"] == "native_runtime"
    assert [item["id"] for item in initial_frame["legal_actions"]["items"][:2]] == [
        "noop",
        "bridge_input",
    ]

    result = env.apply(
        ArenaAction(
            player="player_0",
            move="bridge_input",
            raw=json.dumps(
                {
                    "move": "bridge_input",
                    "payload": {
                        "event_type": "mouse_down",
                        "x": 320,
                        "y": 160,
                        "button": "left",
                    },
                }
            ),
        )
    )

    assert result is None
    assert bridge.submitted_inputs == [
        {
            "event_type": "mouse_down",
            "x": 320,
            "y": 160,
            "button": "left",
        }
    ]


def test_openra_native_env_keeps_local_player_active_between_steps() -> None:
    bridge = _FakeNativeBridge(
        {
            "frame_data_url": _png_data_url(_SMALL_PNG_BYTES),
            "mime_type": "image/png",
            "engine_tick": 16,
            "viewport": {"width": 1280, "height": 720},
        },
        {
            "frame_data_url": _png_data_url(_SMALL_PNG_BYTES),
            "mime_type": "image/png",
            "engine_tick": 17,
            "viewport": {"width": 1280, "height": 720},
        },
    )
    env = OpenRAArenaEnvironment(
        env_id="ra_skirmish_1v1",
        backend_mode="native",
        mod_id="ra",
        map_ref="map/openra/ra/marigold-town.oramap",
        native_bridge=bridge,
        player_ids=("player_0", "player_1"),
    )

    assert env.get_active_player() == "player_0"

    result = env.apply(ArenaAction(player="player_0", move="noop", raw="noop"))

    assert result is None
    assert env.get_active_player() == "player_0"
    refreshed_frame = env.get_last_frame()
    assert refreshed_frame["view"]["image"] == {
        "data_url": _png_data_url(_ALT_SMALL_PNG_BYTES),
        "mimeType": "image/png",
    }
    assert refreshed_frame["tick"] == 18
    assert refreshed_frame["last_move"] == "bridge_input:mouse_down"


def test_openra_native_env_get_last_frame_refreshes_bridge_state_for_live_streaming() -> None:
    bridge = _FakeNativeBridge(
        {
            "frame_data_url": _png_data_url(_SMALL_PNG_BYTES),
            "mime_type": "image/png",
            "engine_tick": 21,
            "viewport": {"width": 1280, "height": 720},
        },
        {
            "frame_data_url": _png_data_url(_ALT_SMALL_PNG_BYTES),
            "mime_type": "image/png",
            "engine_tick": 22,
            "viewport": {"width": 1280, "height": 720},
        },
        {
            "frame_data_url": _png_data_url(_SMALL_PNG_BYTES),
            "mime_type": "image/png",
            "engine_tick": 23,
            "viewport": {"width": 1280, "height": 720},
        },
    )
    env = OpenRAArenaEnvironment(
        env_id="ra_skirmish_1v1",
        backend_mode="native",
        mod_id="ra",
        map_ref="map/openra/ra/marigold-town.oramap",
        native_bridge=bridge,
    )

    first_frame = env.get_last_frame()
    second_frame = env.get_last_frame()

    assert first_frame["tick"] == 22
    assert second_frame["tick"] == 23
    assert first_frame["view"]["image"]["data_url"] != second_frame["view"]["image"]["data_url"]


def test_openra_native_env_from_runtime_prefers_max_decisions_over_stub_ticks(
    monkeypatch,
) -> None:
    captured_bridge_kwargs: dict[str, object] = {}

    def _fake_bridge(**kwargs):
        captured_bridge_kwargs.update(kwargs)
        return _FakeNativeBridge(
            {
                "frame_data_url": _png_data_url(_SMALL_PNG_BYTES),
                "mime_type": "image/png",
                "engine_tick": 21,
                "viewport": {"width": 1280, "height": 720},
            }
        )

    monkeypatch.setattr(openra_env_module, "OpenRANativeBridge", _fake_bridge)

    env = OpenRAArenaEnvironment.from_runtime(
        sample=SimpleNamespace(
            runtime_overrides={
                "backend_mode": "native",
                "max_decisions": 64,
            }
        ),
        resolved=SimpleNamespace(
            game_kit=SimpleNamespace(defaults={"stub_max_ticks": 6}),
            env_spec=SimpleNamespace(
                env_id="ra_skirmish_1v1",
                defaults={
                    "env_id": "ra_skirmish_1v1",
                    "mod_id": "ra",
                    "map_ref": "map/openra/ra/marigold-town.oramap",
                },
            ),
        ),
        resources=None,
        player_specs=[
            SimpleNamespace(player_id="player_0", display_name="Commander"),
        ],
        invocation_context=None,
    )

    assert env._max_decisions == 64
    assert captured_bridge_kwargs["launch_mode"] == "skirmish"


def test_openra_native_env_observe_handles_empty_selection_without_index_error() -> None:
    bridge = _FakeNativeBridge(
        {
            "frame_data_url": _png_data_url(_SMALL_PNG_BYTES),
            "mime_type": "image/png",
            "engine_tick": 30,
            "viewport": {"width": 1280, "height": 720},
        }
    )
    env = OpenRAArenaEnvironment(
        env_id="ra_skirmish_1v1",
        backend_mode="native",
        mod_id="ra",
        map_ref="map/openra/ra/marigold-town.oramap",
        native_bridge=bridge,
    )

    observation = env.observe("player_0")

    assert observation.metadata["selection"] == {
        "unit_ids": [],
        "primary_unit_id": None,
    }


def test_openra_native_env_projects_local_units_from_bridge_state() -> None:
    bridge = _FakeNativeBridge(
        {
            "frame_data_url": _png_data_url(_SMALL_PNG_BYTES),
            "mime_type": "image/png",
            "engine_tick": 34,
            "viewport": {"width": 1280, "height": 720},
            "playable_player_count": 2,
            "local_player_actor_count": 2,
            "camera_center": {"x": 64000, "y": 15872, "z": 0},
        }
    )
    env = OpenRAArenaEnvironment(
        env_id="ra_skirmish_1v1",
        backend_mode="native",
        mod_id="ra",
        map_ref="map/openra/ra/marigold-town.oramap",
        native_bridge=bridge,
    )

    frame = env.get_last_frame()
    units = frame["metadata"]["units"]
    observation = env.observe("player_0")

    assert [unit["id"] for unit in units] == ["native_unit_1", "native_unit_2"]
    assert [unit["owner"] for unit in units] == ["player_0", "player_0"]
    assert [unit["selected"] for unit in units] == [True, False]
    assert frame["metadata"]["selection"] == {
        "unit_ids": ["native_unit_1"],
        "primary_unit_id": "native_unit_1",
    }
    assert units[0]["position"] == {"x": 62, "y": 16}
    assert units[1]["position"] == {"x": 63, "y": 17}
    assert observation.metadata["selection"] == {
        "unit_ids": ["native_unit_1"],
        "primary_unit_id": "native_unit_1",
    }
    assert [unit["id"] for unit in observation.metadata["units"]] == [
        "native_unit_1",
        "native_unit_2",
    ]


def test_openra_native_env_demo_script_turns_noop_into_bridge_inputs() -> None:
    bridge = _FakeNativeBridge(
        {
            "frame_data_url": _png_data_url(_SMALL_PNG_BYTES),
            "mime_type": "image/png",
            "engine_tick": 40,
            "viewport": {"width": 1280, "height": 720},
        }
    )
    env = OpenRAArenaEnvironment(
        env_id="ra_skirmish_1v1",
        backend_mode="native",
        mod_id="ra",
        map_ref="map/openra/ra/marigold-town.oramap",
        native_bridge=bridge,
        native_demo_script="camera_tour",
    )

    result = env.apply(ArenaAction(player="player_0", move="noop", raw="noop"))

    assert result is None
    assert bridge.submitted_inputs == [
        {
            "event_type": "key_down",
            "key": "ArrowRight",
            "modifiers": [],
        }
    ]


def test_openra_native_env_unit_patrol_demo_script_issues_real_mouse_orders() -> None:
    bridge = _FakeNativeBridge(
        {
            "frame_data_url": _png_data_url(_SMALL_PNG_BYTES),
            "mime_type": "image/png",
            "engine_tick": 50,
            "viewport": {"width": 1280, "height": 720},
        }
    )
    env = OpenRAArenaEnvironment(
        env_id="ra_skirmish_1v1",
        backend_mode="native",
        mod_id="ra",
        map_ref="map/openra/ra/marigold-town.oramap",
        native_bridge=bridge,
        max_decisions=8,
        native_demo_script="unit_patrol",
        native_step_interval_s=0.05,
    )

    for _ in range(4):
        result = env.apply(ArenaAction(player="player_0", move="noop", raw="noop"))
        assert result is None

    submitted = [
        (
            str(item.get("event_type")),
            str(item.get("button")),
            int(item.get("x", -1)),
            int(item.get("y", -1)),
        )
        for item in bridge.submitted_inputs
    ]

    assert submitted == [
        ("mouse_down", "left", 724, 380),
        ("mouse_up", "left", 724, 380),
        ("mouse_down", "right", 920, 332),
        ("mouse_up", "right", 920, 332),
    ]


def test_openra_native_env_live_frame_polling_continues_demo_script(monkeypatch) -> None:
    bridge = _FakeNativeBridge(
        {
            "frame_data_url": _png_data_url(_SMALL_PNG_BYTES),
            "mime_type": "image/png",
            "engine_tick": 60,
            "viewport": {"width": 1280, "height": 720},
        }
    )
    env = OpenRAArenaEnvironment(
        env_id="ra_skirmish_1v1",
        backend_mode="native",
        mod_id="ra",
        map_ref="map/openra/ra/marigold-town.oramap",
        native_bridge=bridge,
        native_demo_script="unit_patrol",
        native_demo_script_mode="live_stream",
        native_step_interval_s=0.25,
    )
    monotonic_values = iter((1.0, 1.1, 1.4, 1.7, 2.0))
    monkeypatch.setattr(
        openra_env_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    for _ in range(5):
        env.get_last_frame()

    submitted = [
        (
            str(item.get("event_type")),
            str(item.get("button")),
            int(item.get("x", -1)),
            int(item.get("y", -1)),
        )
        for item in bridge.submitted_inputs
    ]

    assert submitted == [
        ("mouse_down", "left", 724, 380),
        ("mouse_up", "left", 724, 380),
        ("mouse_down", "right", 920, 332),
        ("mouse_up", "right", 920, 332),
    ]


def test_openra_native_env_unit_patrol_avoids_keyboard_camera_events() -> None:
    bridge = _FakeNativeBridge(
        {
            "frame_data_url": _png_data_url(_SMALL_PNG_BYTES),
            "mime_type": "image/png",
            "engine_tick": 70,
            "viewport": {"width": 1280, "height": 720},
        }
    )
    env = OpenRAArenaEnvironment(
        env_id="ra_skirmish_1v1",
        backend_mode="native",
        mod_id="ra",
        map_ref="map/openra/ra/marigold-town.oramap",
        native_bridge=bridge,
        max_decisions=8,
        native_demo_script="unit_patrol",
        native_step_interval_s=0.05,
    )

    for _ in range(6):
        result = env.apply(ArenaAction(player="player_0", move="noop", raw="noop"))
        assert result is None

    assert bridge.submitted_inputs[4] == {
        "event_type": "mouse_down",
        "button": "right",
        "buttons": ["right"],
        "x": 944,
        "y": 520,
        "viewport": {"width": 1280, "height": 720},
        "modifiers": [],
    }
    assert bridge.submitted_inputs[5] == {
        "event_type": "mouse_up",
        "button": "right",
        "buttons": ["right"],
        "x": 944,
        "y": 520,
        "viewport": {"width": 1280, "height": 720},
        "modifiers": [],
    }
    assert not any(
        str(item.get("event_type", "")).startswith("key_")
        for item in bridge.submitted_inputs
    )
