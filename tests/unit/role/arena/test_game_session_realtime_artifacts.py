from __future__ import annotations

from types import SimpleNamespace

import pytest

from gage_eval.game_kits.contracts import RealtimeHumanControlProfile
import gage_eval.role.arena.core.game_session as game_session_module
from gage_eval.role.arena.core.game_session import GameSession
from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.role.arena.visualization.recorder import ArenaVisualSessionRecorder


def test_realtime_snapshot_stride_decimates_persistent_artifacts_without_disabling_live_frames() -> None:
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.retro.frame_v1",
        game_id="retro_mario",
        scheduling_family="real_time_tick",
        session_id="sample-1",
    )
    control = RealtimeHumanControlProfile(
        mode="scheduler_owned_human_realtime",
        activation_scope="pure_human_only",
        input_model="continuous_state",
        tick_interval_ms=16,
        input_transport="realtime_ws",
        frame_output_hz=60,
        artifact_sampling_mode="async_decimated_live",
        snapshot_persist_stride=3,
    )
    session = GameSession(
        sample=ArenaSample(
            game_kit="retro_platformer",
            env="retro_mario",
            scheduler="real_time_tick/default",
        ),
        runtime_profile=SimpleNamespace(realtime_human_control=control),
        visual_recorder=recorder,
    )

    session._last_realtime_snapshot_ts_ms = 100  # noqa: SLF001
    session.tick = 0
    assert session._should_record_realtime_snapshot() is True  # noqa: SLF001

    session.tick = 1
    assert session._should_record_realtime_snapshot() is False  # noqa: SLF001

    session.tick = 3
    assert session._should_record_realtime_snapshot() is True  # noqa: SLF001

    session.tick = 6
    assert session._should_record_realtime_snapshot() is True  # noqa: SLF001

    session.tick = 4
    session.final_result = object()
    assert session._should_record_realtime_snapshot() is True  # noqa: SLF001


def test_realtime_snapshot_stride_one_records_every_tick() -> None:
    control = RealtimeHumanControlProfile(
        mode="scheduler_owned_human_realtime",
        activation_scope="pure_human_only",
        input_model="continuous_state",
        tick_interval_ms=16,
        input_transport="realtime_ws",
        artifact_sampling_mode="async_decimated_live",
        snapshot_persist_stride=1,
    )
    session = GameSession(
        sample=ArenaSample(
            game_kit="retro_platformer",
            env="retro_mario",
            scheduler="real_time_tick/default",
        ),
        runtime_profile=SimpleNamespace(realtime_human_control=control),
        visual_recorder=ArenaVisualSessionRecorder(
            plugin_id="arena.visualization.retro.frame_v1",
            game_id="retro_mario",
            scheduling_family="real_time_tick",
            session_id="sample-1",
        ),
    )

    session._last_realtime_snapshot_ts_ms = 100  # noqa: SLF001
    session.tick = 1

    assert session._should_record_realtime_snapshot() is True  # noqa: SLF001


def test_frame_snapshot_encoder_prefers_visual_webp_and_reuses_buffer(monkeypatch) -> None:
    save_calls: list[dict[str, object]] = []

    class FakeImage:
        mode = "RGB"

        def save(self, buffer, **kwargs):  # noqa: ANN001
            save_calls.append(dict(kwargs))
            buffer.write(b"webp-bytes")

    class FakeImageModule:
        @staticmethod
        def fromarray(frame):  # noqa: ANN001
            return FakeImage()

    monkeypatch.setattr(game_session_module, "Image", FakeImageModule)

    first_url = game_session_module._encode_frame_data_url(object())  # noqa: SLF001
    second_url = game_session_module._encode_frame_data_url(object())  # noqa: SLF001

    assert first_url == "data:image/webp;base64,d2VicC1ieXRlcw=="
    assert second_url == first_url
    assert save_calls == [
        {"format": "WEBP", "lossless": False, "quality": 95},
        {"format": "WEBP", "lossless": False, "quality": 95},
    ]


def test_frame_snapshot_encoder_warns_once_when_webp_falls_back_to_png(monkeypatch) -> None:
    warnings: list[str] = []

    class FakeLogger:
        @staticmethod
        def warning(message: str) -> None:
            warnings.append(message)

    class FakeImage:
        mode = "RGB"

        def save(self, buffer, **kwargs):  # noqa: ANN001
            if kwargs.get("format") == "WEBP":
                raise OSError("encoder webp not available")
            buffer.write(b"png-bytes")

    class FakeImageModule:
        @staticmethod
        def fromarray(frame):  # noqa: ANN001
            return FakeImage()

    monkeypatch.setattr(game_session_module, "Image", FakeImageModule)
    monkeypatch.setattr(game_session_module, "logger", FakeLogger())
    monkeypatch.setattr(game_session_module, "_WEBP_FALLBACK_LOGGED", False)

    first_url = game_session_module._encode_frame_data_url(object())  # noqa: SLF001
    second_url = game_session_module._encode_frame_data_url(object())  # noqa: SLF001

    assert first_url == "data:image/png;base64,cG5nLWJ5dGVz"
    assert second_url == first_url
    assert warnings == [
        "WebP encoder unavailable; falling back to PNG. Install libwebp for better realtime frame encoding."
    ]


def test_frame_snapshot_encoder_reuses_pooled_numpy_frame_arrays(monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    frames_seen: list[object] = []

    class FakeImage:
        mode = "RGB"

        def save(self, buffer, **kwargs):  # noqa: ANN001, ARG002
            buffer.write(b"webp-bytes")

    class FakeImageModule:
        @staticmethod
        def fromarray(frame):  # noqa: ANN001
            frames_seen.append(frame)
            return FakeImage()

    monkeypatch.setattr(game_session_module, "Image", FakeImageModule)
    game_session_module._FRAME_ARRAY_POOL.clear()  # noqa: SLF001

    first_source = np.zeros((2, 3, 3), dtype=np.uint8)
    second_source = np.ones((2, 3, 3), dtype=np.uint8)

    game_session_module._encode_frame_data_url(first_source)  # noqa: SLF001
    game_session_module._encode_frame_data_url(second_source)  # noqa: SLF001

    assert len(frames_seen) == 2
    assert frames_seen[0] is frames_seen[1]
    assert frames_seen[0] is not first_source
    assert frames_seen[1] is not second_source
