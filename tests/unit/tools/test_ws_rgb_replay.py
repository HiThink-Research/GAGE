from __future__ import annotations

import json
from pathlib import Path

from gage_eval.tools import ws_rgb_replay


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_build_replay_v1_display_prefers_frame_events(tmp_path: Path) -> None:
    replay_dir = tmp_path / "runs" / "run_demo" / "replays" / "sample_1"
    frame_path = replay_dir / "frames" / "frame_000000.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_bytes(b"\xff\xd8\xff\xd9")

    events_path = replay_dir / "events.jsonl"
    events_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "action", "seq": 1, "step": 1, "actor": "player_0", "move": "A1"}),
                json.dumps(
                    {
                        "type": "frame",
                        "seq": 2,
                        "step": 1,
                        "actor": "player_0",
                        "frame": {"board_text": "frame-1"},
                        "image": {"path": "frames/frame_000000.jpg", "format": "jpeg", "width": 4, "height": 4},
                    }
                ),
                json.dumps({"type": "result", "seq": 3, "result": "draw"}),
            ]
        ),
        encoding="utf-8",
    )
    replay_manifest = replay_dir / "replay.json"
    _write_json(
        replay_manifest,
        {
            "schema": "gage_replay/v1",
            "recording": {"events_path": "events.jsonl"},
            "meta": {"scheduler_type": "turn"},
        },
    )

    sample_record = {
        "task_id": "pettingzoo_pong_dummy_ws_rgb",
        "sample": {
            "id": "sample_1",
            "metadata": {"player_ids": ["player_0"]},
            "predict_result": [{"replay_path": str(replay_manifest)}],
        },
    }
    display = ws_rgb_replay._build_replay_v1_display(
        sample_record,
        task_id="pettingzoo_pong_dummy_ws_rgb",
        fps=10.0,
        max_frames=0,
    )

    assert display is not None
    frame_source = display["frame_source"]
    frame = frame_source()
    assert frame["board_text"] == "frame-1"
    assert frame["_image_path_abs"] == str(frame_path.resolve())
    frame_count = display["frame_count"]
    frame_at = display["frame_at"]
    assert callable(frame_count)
    assert callable(frame_at)
    assert frame_count() == 1
    assert frame_at(0)["board_text"] == "frame-1"


def test_build_replay_v1_display_returns_none_without_frame_events(tmp_path: Path) -> None:
    replay_dir = tmp_path / "runs" / "run_demo" / "replays" / "sample_2"
    replay_dir.mkdir(parents=True, exist_ok=True)
    events_path = replay_dir / "events.jsonl"
    events_path.write_text(json.dumps({"type": "action", "seq": 1}), encoding="utf-8")

    replay_manifest = replay_dir / "replay.json"
    _write_json(
        replay_manifest,
        {
            "schema": "gage_replay/v1",
            "recording": {"events_path": "events.jsonl"},
            "meta": {"scheduler_type": "turn"},
        },
    )
    sample_record = {
        "sample": {
            "id": "sample_2",
            "predict_result": [{"replay_path": str(replay_manifest)}],
        }
    }

    display = ws_rgb_replay._build_replay_v1_display(
        sample_record,
        task_id="task_no_frame",
        fps=10.0,
        max_frames=0,
    )

    assert display is None
