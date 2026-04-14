from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import gage_eval.role.arena.visualization.recorder as recorder_module
from gage_eval.role.arena.types import GameResult
from gage_eval.role.arena.visualization.contracts import ControlCommand
from gage_eval.role.arena.visualization.recorder import ArenaVisualSessionRecorder
from gage_eval.role.arena.visualization.artifacts import to_bounded_json_safe, to_visual_json_safe


def test_visual_session_recorder_persists_timeline_manifest_snapshot_and_markers(tmp_path: Path) -> None:
    replay_path = tmp_path / "runs" / "run-1" / "replays" / "sample-1" / "replay.json"
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena-role",
        game_id="gomoku",
        scheduling_family="turn",
        session_id="sample-1",
    )

    recorder.record_decision_window_open(
        ts_ms=1001,
        step=1,
        tick=0,
        player_id="player_0",
        observation={"board_text": "demo"},
    )
    recorder.record_action_intent(
        ts_ms=1002,
        step=1,
        tick=0,
        player_id="player_0",
        action={"move": "A1"},
    )
    recorder.record_action_committed(
        ts_ms=1003,
        step=1,
        tick=0,
        player_id="player_0",
        action={"move": "A1"},
    )
    recorder.record_decision_window_close(
        ts_ms=1004,
        step=1,
        tick=1,
        player_id="player_0",
    )
    recorder.record_snapshot(
        ts_ms=1005,
        step=2,
        tick=1,
        snapshot={"board_text": "demo-board"},
    )
    recorder.record_result(
        ts_ms=1006,
        step=2,
        tick=1,
        result=GameResult(
            winner="player_0",
            result="win",
            reason="terminal",
            move_count=1,
            illegal_move_count=0,
            final_board="demo-board",
            move_log=[],
            replay_path=str(replay_path),
        ),
    )

    assert recorder.build_visual_session().lifecycle == "live_ended"

    artifacts = recorder.persist(replay_path)

    assert artifacts.visual_session_ref == str(
        replay_path.parent / "arena_visual_session" / "v1" / "manifest.json"
    )
    assert artifacts.manifest_path.exists()
    assert artifacts.timeline_path.exists()
    assert artifacts.index_path.exists()
    assert artifacts.snapshot_anchors

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["visualSession"]["pluginId"] == "arena-role"
    assert manifest["visualSession"]["gameId"] == "gomoku"
    assert manifest["visualSession"]["scheduling"]["family"] == "turn"
    assert manifest["visualSession"]["timeline"]["eventCount"] == 6

    timeline_lines = artifacts.timeline_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["type"] for line in timeline_lines] == [
        "decision_window_open",
        "action_intent",
        "action_committed",
        "decision_window_close",
        "snapshot",
        "result",
    ]
    assert [json.loads(line)["seq"] for line in timeline_lines] == [1, 2, 3, 4, 5, 6]

    snapshot_path = Path(artifacts.snapshot_anchors[0]["snapshotRef"])
    assert snapshot_path.exists()
    snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot_payload["anchor"] is True
    assert snapshot_payload["seq"] == 5


def test_visual_session_recorder_can_mark_non_human_decision_windows_as_non_interactive() -> None:
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.doudizhu.table_v1",
        game_id="doudizhu",
        scheduling_family="turn",
        session_id="sample-1",
    )

    recorder.record_decision_window_open(
        ts_ms=1001,
        step=1,
        tick=1,
        player_id="farmer_right",
        observation={"legal_moves": ["pass"]},
        accepts_human_intent=False,
    )

    session = recorder.build_visual_session()

    assert session.scheduling.phase == "waiting_for_intent"
    assert session.scheduling.accepts_human_intent is False
    assert session.scheduling.active_actor_id == "farmer_right"


def test_visual_session_recorder_can_update_scheduling_state_without_timeline_events() -> None:
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.retro.frame_v1",
        game_id="retro_mario",
        scheduling_family="real_time_tick",
        session_id="sample-1",
    )

    recorder.set_scheduling_state(
        phase="waiting_for_intent",
        accepts_human_intent=True,
        active_actor_id="player_0",
    )
    recorder.update_runtime_metrics(tick_overshoot_ms=3.5, artifact_queue_depth=0)

    session = recorder.build_visual_session()
    header = recorder.export_live_header()

    assert session.scheduling.phase == "waiting_for_intent"
    assert session.scheduling.accepts_human_intent is True
    assert session.summary["realtimeMetrics"]["tick_overshoot_ms"] == 3.5
    assert header["lifecycle"] == "initializing"
    assert header["tailSeq"] == 0


def test_visual_session_recorder_drains_enqueued_snapshots_outside_tick_path() -> None:
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.retro.frame_v1",
        game_id="retro_mario",
        scheduling_family="real_time_tick",
        session_id="sample-async-artifacts",
    )

    recorder.enqueue_snapshot(
        ts_ms=1010,
        step=4,
        tick=4,
        snapshot={"board_text": "queued-frame"},
        snapshot_is_scene_safe=True,
    )

    assert recorder.pending_snapshot_count() == 1
    assert recorder.export_live_state().timeline_events == ()

    header = recorder.export_live_header()

    assert header["tailSeq"] == 1
    assert recorder.pending_snapshot_count() == 0
    live_state = recorder.export_live_state()
    assert [event.type for event in live_state.timeline_events] == ["snapshot"]
    assert live_state.snapshot_payloads[0]["snapshot"]["board_text"] == "queued-frame"


def test_visual_session_recorder_can_drain_snapshots_in_background() -> None:
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.retro.frame_v1",
        game_id="retro_mario",
        scheduling_family="real_time_tick",
        session_id="sample-bg-artifacts",
    )

    recorder.start_background_snapshot_drain(poll_interval_s=0.001)
    try:
        recorder.enqueue_snapshot(
            ts_ms=1010,
            step=4,
            tick=4,
            snapshot={"board_text": "queued-frame"},
            snapshot_is_scene_safe=True,
        )

        deadline = time.monotonic() + 1.0
        while recorder.pending_snapshot_count() > 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        live_state = recorder.export_live_state()

        assert recorder.pending_snapshot_count() == 0
        assert [event.type for event in live_state.timeline_events] == ["snapshot"]
        assert live_state.snapshot_payloads[0]["snapshot"]["board_text"] == "queued-frame"
    finally:
        recorder.stop_background_snapshot_drain()


def test_visual_session_recorder_drops_oldest_pending_snapshot_when_async_queue_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[tuple[str, int]] = []

    class FakeLogger:
        @staticmethod
        def warning(message: str, total: int) -> None:
            warnings.append((message, total))

    monkeypatch.setattr(recorder_module, "logger", FakeLogger())
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.retro.frame_v1",
        game_id="retro_mario",
        scheduling_family="real_time_tick",
        session_id="sample-bg-artifacts",
    )
    recorder._max_pending_snapshots = 2  # noqa: SLF001

    for tick in range(1, 4):
        recorder.enqueue_snapshot(
            ts_ms=1000 + tick,
            step=tick,
            tick=tick,
            snapshot={"board_text": f"queued-frame-{tick}"},
            snapshot_is_scene_safe=True,
        )

    assert recorder.pending_snapshot_count() == 2
    recorder.flush_pending_snapshots()
    live_state = recorder.export_live_state()

    assert [
        snapshot["snapshot"]["board_text"]
        for snapshot in live_state.snapshot_payloads
    ] == ["queued-frame-2", "queued-frame-3"]
    assert recorder.build_visual_session().summary["realtimeMetrics"]["dropped_snapshot_count"] == 1
    assert warnings == [
        ("ArenaVisualSessionRecorder dropped pending snapshot, total={}", 1)
    ]


def test_visual_session_recorder_persists_seek_snapshot_index(tmp_path: Path) -> None:
    replay_path = tmp_path / "runs" / "run-1" / "replays" / "sample-1" / "replay.json"
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena-role",
        game_id="gomoku",
        scheduling_family="turn",
        session_id="sample-1",
    )

    recorder.record_snapshot(
        ts_ms=1005,
        step=2,
        tick=1,
        snapshot={"board_text": "demo-board"},
        anchor=True,
    )

    artifacts = recorder.persist(replay_path)

    seek_index = json.loads((artifacts.layout.session_dir / "seek_snapshots.json").read_text(encoding="utf-8"))
    assert seek_index["seekSnapshots"] == [
        {
            "seq": 1,
            "tsMs": 1005,
            "snapshotMode": "full",
            "snapshotRef": "snapshots/seq-000001.json",
        }
    ]


@pytest.mark.parametrize(
    ("visual_kind", "expected_snapshot_mode"),
    [
        ("board", "full"),
        ("table", "full"),
        ("frame", "media_ref"),
    ],
)
def test_visual_session_recorder_applies_seek_snapshot_mode_by_visual_kind(
    tmp_path: Path,
    visual_kind: str,
    expected_snapshot_mode: str,
) -> None:
    replay_path = tmp_path / "runs" / visual_kind / "replays" / "sample-1" / "replay.json"
    recorder = ArenaVisualSessionRecorder(
        plugin_id=f"arena.visualization.{visual_kind}.v1",
        game_id="demo",
        scheduling_family="turn",
        session_id="sample-1",
        visual_kind=visual_kind,
    )

    recorder.record_snapshot(
        ts_ms=1005,
        step=2,
        tick=1,
        snapshot={
            "board": {"state": "demo"} if visual_kind != "frame" else None,
            "media": {
                "primary": {
                    "mediaId": "frame-1",
                    "transport": "artifact_ref",
                    "mimeType": "image/png",
                    "url": "frames/frame-1.png",
                }
            }
            if visual_kind == "frame"
            else None,
        },
        anchor=True,
    )

    artifacts = recorder.persist(replay_path)

    seek_index = json.loads((artifacts.layout.session_dir / "seek_snapshots.json").read_text(encoding="utf-8"))
    assert seek_index["seekSnapshots"][0]["snapshotMode"] == expected_snapshot_mode


def test_visual_session_recorder_bounds_heavy_snapshot_payloads() -> None:
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena-role",
        game_id="gomoku",
        scheduling_family="turn",
        session_id="sample-1",
    )
    payload = {
        "board_text": "demo-board",
        "raw_obs": {
            "_rgb": b"\x00" * 2048,
            "frame": {"pixels": [1, 2, 3], "dtype": "uint8"},
        },
        "message": "x" * 800,
        "nested": {
            "frames": [{"blob": b"\x01" * 1024}],
            "note": "lightweight",
        },
    }

    snapshot = recorder.record_snapshot(
        ts_ms=2001,
        step=1,
        tick=1,
        snapshot=payload,
    )

    bounded = snapshot.payload["snapshot"]
    assert bounded["board_text"] == "demo-board"
    assert bounded["raw_obs"]["kind"] == "mapping"
    assert bounded["raw_obs"]["size"] == 2
    assert bounded["message"] == payload["message"]
    assert bounded["nested"]["frames"][0]["blob"]["kind"] == "bytes"
    assert bounded["nested"]["frames"][0]["blob"]["size"] == 1024
    assert to_bounded_json_safe(payload)["message"].endswith("...<len=800>")
    assert to_bounded_json_safe(payload)["raw_obs"]["kind"] == "mapping"


def test_to_visual_json_safe_preserves_inline_data_urls() -> None:
    data_url = "data:image/png;base64," + ("QUJD" * 3000)
    payload = {
        "view": {
            "image": {
                "data_url": data_url,
                "data": "x" * 12000,
                "shape": [240, 256, 3],
            }
        }
    }

    bounded = to_visual_json_safe(payload)

    assert bounded["view"]["image"]["data_url"] == data_url
    assert bounded["view"]["image"]["data"]["kind"] == "string"
    assert bounded["view"]["image"]["data"]["size"] == 12000


def test_visual_session_recorder_steps_replay_across_stable_turn_checkpoints() -> None:
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.tictactoe.board_v1",
        game_id="tictactoe",
        scheduling_family="turn",
        session_id="sample-1",
    )

    recorder.record_decision_window_open(
        ts_ms=1001,
        step=0,
        tick=0,
        player_id="X",
        observation={"board_text": "empty"},
    )
    recorder.record_action_intent(
        ts_ms=1002,
        step=0,
        tick=0,
        player_id="X",
        action={"move": "1,1"},
    )
    recorder.record_action_committed(
        ts_ms=1003,
        step=0,
        tick=0,
        player_id="X",
        action={"move": "1,1"},
    )
    recorder.record_decision_window_close(
        ts_ms=1004,
        step=0,
        tick=0,
        player_id="X",
    )
    recorder.record_snapshot(
        ts_ms=1005,
        step=1,
        tick=1,
        snapshot={"board_text": "stale-after-x"},
    )
    recorder.record_decision_window_open(
        ts_ms=1006,
        step=1,
        tick=1,
        player_id="O",
        observation={"board_text": "x-applied"},
    )
    recorder.record_action_intent(
        ts_ms=1007,
        step=1,
        tick=1,
        player_id="O",
        action={"move": "1,2"},
    )
    recorder.record_action_committed(
        ts_ms=1008,
        step=1,
        tick=1,
        player_id="O",
        action={"move": "1,2"},
    )
    recorder.record_decision_window_close(
        ts_ms=1009,
        step=1,
        tick=1,
        player_id="O",
    )
    recorder.record_snapshot(
        ts_ms=1010,
        step=2,
        tick=2,
        snapshot={"board_text": "stale-after-o"},
    )
    recorder.record_result(
        ts_ms=1011,
        step=2,
        tick=2,
        result=GameResult(
            winner="X",
            result="win",
            reason="three_in_row",
            move_count=2,
            illegal_move_count=0,
            final_board="x-o-final",
            move_log=[],
        ),
    )

    replay_seq = recorder.apply_control_command(ControlCommand(command_type="replay"))
    assert replay_seq == 1

    step_one = recorder.apply_control_command(
        ControlCommand(command_type="step", step_delta=1)
    )
    assert step_one == 6

    step_two = recorder.apply_control_command(
        ControlCommand(command_type="step", step_delta=1)
    )
    assert step_two == 11

    step_three = recorder.apply_control_command(
        ControlCommand(command_type="step", step_delta=1)
    )
    assert step_three == 11
