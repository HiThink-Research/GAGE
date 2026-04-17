from __future__ import annotations

from gage_eval.game_kits.aec_env_game.pettingzoo.visualization import (
    VISUALIZATION_SPEC as PETTINGZOO_VISUALIZATION_SPEC,
)
from gage_eval.role.arena.visualization.live_session import RecorderLiveSessionSource
from gage_eval.role.arena.visualization.recorder import ArenaVisualSessionRecorder


def test_low_latency_stream_frame_uses_spec_default_stream_id_when_payload_omits_stream_id() -> None:
    recorder = ArenaVisualSessionRecorder(
        plugin_id=PETTINGZOO_VISUALIZATION_SPEC.plugin_id,
        game_id="pettingzoo",
        scheduling_family="real_time_tick",
        session_id="sample-pettingzoo-low-latency",
    )

    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-pettingzoo-low-latency",
        visualization_spec=PETTINGZOO_VISUALIZATION_SPEC,
        live_scene_scheme="low_latency_channel",
        live_frame_supplier=lambda: {
            "media": {
                "primary": {
                    "mediaId": "pettingzoo-frame-live",
                    "transport": "http_pull",
                    "mimeType": "image/png",
                    "url": "data:image/png;base64,YmFy",
                }
            },
        },
    )

    assert live_source.load_stream_frame("live-channel-arena") == (b"bar", "image/png")


def test_low_latency_scene_cache_ignores_stream_frame_version(
    monkeypatch,
) -> None:
    recorder = ArenaVisualSessionRecorder(
        plugin_id=PETTINGZOO_VISUALIZATION_SPEC.plugin_id,
        game_id="pettingzoo",
        scheduling_family="real_time_tick",
        session_id="sample-pettingzoo-low-latency-cache",
    )
    event = recorder.record_snapshot(
        ts_ms=1001,
        step=1,
        tick=1,
        snapshot={
            "activePlayerId": "player_0",
            "board_text": "cached scene",
            "legalActions": [],
        },
        snapshot_is_scene_safe=True,
    )
    frame_version = {"value": 1}
    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-pettingzoo-low-latency-cache",
        visualization_spec=PETTINGZOO_VISUALIZATION_SPEC,
        live_scene_scheme="low_latency_channel",
        live_frame_supplier=lambda: {
            "_live_frame_version": frame_version["value"],
            "stream_id": "arena",
        },
    )
    export_calls = []
    original_export_live_state = ArenaVisualSessionRecorder.export_live_state

    def counted_export_live_state(self):
        export_calls.append(1)
        return original_export_live_state(self)

    monkeypatch.setattr(
        ArenaVisualSessionRecorder,
        "export_live_state",
        counted_export_live_state,
    )

    first_scene = live_source.load_scene(seq=event.seq)
    frame_version["value"] = 2
    second_scene = live_source.load_scene(seq=event.seq)

    assert first_scene is not None
    assert second_scene is first_scene
    assert len(export_calls) == 1
