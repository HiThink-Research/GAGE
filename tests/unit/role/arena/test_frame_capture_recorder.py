from __future__ import annotations

import json
from pathlib import Path

import pytest

from gage_eval.role.arena.frame_capture import FrameCaptureRecorder


def test_frame_capture_recorder_writes_jpeg_and_frame_event(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")

    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    recorder = FrameCaptureRecorder(
        replay_dir=tmp_path,
        enabled=True,
        frame_stride=1,
        max_frames=0,
        image_format="jpeg",
        jpeg_quality=70,
    )

    recorder.capture(
        {
            "active_player_id": "player_0",
            "board_text": "demo",
            "_rgb": frame,
        },
        step=3,
        actor="player_0",
    )

    events = recorder.build_frame_events()
    assert len(events) == 1
    event = events[0]
    assert event["type"] == "frame"
    assert event["step"] == 3
    assert event["actor"] == "player_0"
    assert "_rgb" not in event["frame"]
    assert event["image"]["path"].startswith("frames/")
    image_path = tmp_path / event["image"]["path"]
    assert image_path.exists()
    assert image_path.stat().st_size > 0
    # Ensure serialized event remains JSON-compatible.
    json.dumps(event, ensure_ascii=False)


def test_frame_capture_recorder_respects_stride_and_max_frames(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    recorder = FrameCaptureRecorder(
        replay_dir=tmp_path,
        enabled=True,
        frame_stride=2,
        max_frames=1,
        image_format="jpeg",
        jpeg_quality=80,
    )

    for step in range(1, 6):
        recorder.capture({"board_text": f"step-{step}", "_rgb": frame}, step=step, actor="player_0")

    events = recorder.build_frame_events()
    assert len(events) == 3
    assert [event["step"] for event in events] == [1, 3, 5]
    assert "image" in events[0]
    assert "image" not in events[1]
    assert events[1]["meta"]["image_skipped"] == "max_frames_reached"
    assert "image" not in events[2]
    assert events[2]["meta"]["image_skipped"] == "max_frames_reached"
