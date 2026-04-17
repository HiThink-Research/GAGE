from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from gage_eval.game_kits.contracts import RealtimeHumanControlProfile, ResolvedRuntimeProfile
import gage_eval.role.arena.core.game_session as game_session_module
from gage_eval.role.arena.core.game_session import GameSession
from gage_eval.role.arena.core.players import PlayerBindingSpec
from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.role.arena.types import GameResult
from gage_eval.role.arena.visualization.recorder import ArenaVisualSessionRecorder


def _build_scheduler_owned_runtime_profile(
    *,
    fallback_move: str | None = "noop",
) -> ResolvedRuntimeProfile:
    return ResolvedRuntimeProfile(
        scheduler_binding="real_time_tick/default",
        scheduler_family="real_time_tick",
        tick_interval_ms=16,
        pure_human_realtime=True,
        scheduler_owns_realtime_clock=True,
        supports_low_latency_realtime_input=True,
        supports_realtime_input_websocket=True,
        realtime_human_control=RealtimeHumanControlProfile(
            mode="scheduler_owned_human_realtime",
            activation_scope="pure_human_only",
            input_model="continuous_state",
            tick_interval_ms=16,
            input_transport="realtime_ws",
            fallback_move=fallback_move,
        ),
    )


def test_tick_realtime_idle_skips_when_env_lacks_tick_idle() -> None:
    session = GameSession(
        sample=ArenaSample(game_kit="retro_platformer", env="retro_mario"),
        environment=SimpleNamespace(),
        runtime_profile=_build_scheduler_owned_runtime_profile(),
    )

    assert session.tick_realtime_idle(frames=1) is False


def test_tick_realtime_idle_propagates_terminal_result() -> None:
    final_result = GameResult(
        winner=None,
        result="draw",
        reason="terminated",
        move_count=0,
        illegal_move_count=0,
        final_board="{}",
        move_log=[],
    )

    class FakeEnvironment:
        def tick_idle(self, *, frames: int, move: str) -> GameResult:
            assert frames == 2
            assert move == "noop"
            return final_result

    session = GameSession(
        sample=ArenaSample(game_kit="retro_platformer", env="retro_mario"),
        environment=FakeEnvironment(),
        runtime_profile=_build_scheduler_owned_runtime_profile(),
    )

    assert session.tick_realtime_idle(frames=2) is True
    assert session.final_result is final_result


def test_tick_realtime_idle_uses_configured_fallback_move() -> None:
    class FakeEnvironment:
        def __init__(self) -> None:
            self.calls: list[tuple[int, str]] = []

        def tick_idle(self, *, frames: int, move: str) -> None:
            self.calls.append((frames, move))
            return None

    environment = FakeEnvironment()
    session = GameSession(
        sample=ArenaSample(game_kit="retro_platformer", env="retro_mario"),
        environment=environment,
        runtime_profile=_build_scheduler_owned_runtime_profile(fallback_move="wait"),
    )

    assert session.tick_realtime_idle(frames=7) is True
    assert environment.calls == [(7, "wait")]


def test_realtime_driver_params_include_configured_command_stale_window() -> None:
    bindings = (
        PlayerBindingSpec(
            seat="player_0",
            player_id="player_0",
            player_kind="human",
            driver_id="player_driver/human_local_input",
            driver_params={},
        ),
    )
    runtime_profile = _build_scheduler_owned_runtime_profile()
    assert runtime_profile.realtime_human_control is not None
    runtime_profile = replace(
        runtime_profile,
        realtime_human_control=replace(
            runtime_profile.realtime_human_control,
            command_stale_after_ms=200,
        ),
    )

    (updated,) = game_session_module._inject_realtime_driver_params(  # noqa: SLF001
        bindings,
        runtime_profile=runtime_profile,
    )

    assert updated.driver_params["scheduler_owned_realtime"] is True
    assert updated.driver_params["command_stale_after_ms"] == 200


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


def test_inline_snapshot_media_stride_accounts_for_persisted_snapshot_stride() -> None:
    control = RealtimeHumanControlProfile(
        mode="scheduler_owned_human_realtime",
        activation_scope="pure_human_only",
        input_model="queued_command",
        tick_interval_ms=33,
        input_transport="realtime_ws",
        frame_output_hz=30,
        artifact_sampling_mode="async_decimated_live",
        snapshot_persist_stride=3,
    )
    session = GameSession(
        sample=ArenaSample(
            game_kit="gymnasium_atari",
            env="space_invaders",
            scheduler="real_time_tick/default",
        ),
        runtime_profile=ResolvedRuntimeProfile(
            scheduler_binding="real_time_tick/default",
            scheduler_family="real_time_tick",
            tick_interval_ms=33,
            pure_human_realtime=True,
            scheduler_owns_realtime_clock=True,
            supports_low_latency_realtime_input=True,
            supports_realtime_input_websocket=False,
            realtime_human_control=control,
        ),
        visual_recorder=ArenaVisualSessionRecorder(
            plugin_id="arena.visualization.pettingzoo.frame_v1",
            game_id="gymnasium_atari",
            scheduling_family="real_time_tick",
            session_id="sample-1",
        ),
    )
    session._visualization_mode = "arena_visual"  # noqa: SLF001
    session._visualization_live_scene_scheme = "low_latency_channel"  # noqa: SLF001

    session.tick = 3
    assert session._should_include_inline_snapshot_media() is False  # noqa: SLF001
    session.tick = 6
    assert session._should_include_inline_snapshot_media() is False  # noqa: SLF001
    session.tick = 9
    assert session._should_include_inline_snapshot_media() is True  # noqa: SLF001
    session.tick = 18
    assert session._should_include_inline_snapshot_media() is True  # noqa: SLF001


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
