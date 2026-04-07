from __future__ import annotations

from types import SimpleNamespace

import pytest

from gage_eval.role.arena.core.game_session import GameSession
from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.role.arena.visualization.recorder import ArenaVisualSessionRecorder
from gage_eval.game_kits.contracts import (
    HumanRealtimeInputProfile,
    RealtimeHumanControlProfile,
    ResolvedRuntimeProfile,
)


def test_game_session_realtime_live_snapshots_throttle_inline_frame_images_for_queued_command_idle_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEnvironment:
        def get_active_player(self) -> str:
            return "player_0"

        def observe(self, player_id: str) -> object:
            return SimpleNamespace(
                active_player=player_id,
                legal_actions_items=("noop", "issue_command"),
                view_text="live frame state",
                board_text="live frame state",
            )

        def consume_session_progress_delta(self) -> int:
            return 1

        def is_terminal(self) -> bool:
            return False

        def get_last_frame(self) -> dict[str, object]:
            return {
                "board_text": "live frame state",
                "stream_id": "main",
                "metadata": {"stream_id": "main"},
                "view": {
                    "text": "live frame state",
                    "image": {
                        "data_url": "data:image/png;base64,Zm9v",
                        "mimeType": "image/png",
                    },
                },
            }

    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.vizdoom.frame_v1",
        game_id="vizdoom",
        scheduling_family="real_time_tick",
        session_id="sample-realtime-live-idle-inline-media",
        visual_kind="frame",
    )
    session = GameSession(
        sample=ArenaSample(game_kit="sample_realtime", env="queued_command"),
        environment=FakeEnvironment(),
        visual_recorder=recorder,
        runtime_profile=ResolvedRuntimeProfile(
            scheduler_binding="real_time_tick/default",
            scheduler_family="real_time_tick",
            tick_interval_ms=50,
            pure_human_realtime=True,
            scheduler_owns_realtime_clock=True,
            supports_low_latency_realtime_input=True,
            supports_realtime_input_websocket=True,
            human_realtime_inputs=(
                HumanRealtimeInputProfile(
                    player_id="player_0",
                    semantics="queued_command",
                    tick_interval_ms=50,
                ),
            ),
            realtime_human_control=RealtimeHumanControlProfile(
                mode="scheduler_owned_human_realtime",
                activation_scope="pure_human_only",
                input_model="queued_command",
                tick_interval_ms=50,
                input_transport="realtime_ws",
                frame_output_hz=20,
                artifact_sampling_mode="async_decimated_live",
                fallback_move="noop",
            ),
        ),
    )
    session._visualization_mode = "arena_visual"  # noqa: SLF001
    session._visualization_live_scene_scheme = "low_latency_channel"  # noqa: SLF001

    current_ts = {"value": 1000}
    monkeypatch.setattr(
        "gage_eval.role.arena.core.game_session.wall_clock_ms",
        lambda: current_ts["value"],
    )

    session.observe()
    session.advance()
    session.capture_output_tick()
    current_ts["value"] = 1055
    session.observe()
    session.advance()
    session.capture_output_tick()

    recorder.export_live_header()
    snapshots = recorder.export_live_state().snapshot_payloads
    first_observation = snapshots[0]["snapshot"]["observation"]
    second_observation = snapshots[1]["snapshot"]["observation"]

    assert first_observation["view"]["image"]["data_url"] == "data:image/png;base64,Zm9v"
    assert "image" not in second_observation["view"]
