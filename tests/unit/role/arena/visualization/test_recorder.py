from __future__ import annotations

import json
from pathlib import Path

from gage_eval.role.arena.types import GameResult
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
    assert bounded["message"].endswith("...<len=800>")
    assert bounded["nested"]["frames"]["kind"] == "sequence"
    assert bounded["nested"]["frames"]["size"] == 1
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
