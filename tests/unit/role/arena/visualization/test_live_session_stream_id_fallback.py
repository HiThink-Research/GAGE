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
