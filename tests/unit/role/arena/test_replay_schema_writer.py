from __future__ import annotations

import json
from pathlib import Path

from gage_eval.role.arena.replay_schema_writer import ReplaySchemaWriter
from gage_eval.role.arena.types import GameResult


def _build_result(*, replay_path: str | None = None) -> GameResult:
    return GameResult(
        winner="player_0",
        result="win",
        reason="terminal",
        move_count=2,
        illegal_move_count=0,
        final_board="{}",
        move_log=[],
        replay_path=replay_path,
    )


def test_replay_schema_writer_writes_manifest_and_events(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run_1"
    writer = ReplaySchemaWriter(run_dir=run_dir, sample_id="sample 1")
    result = _build_result(replay_path="legacy.json")
    move_log = [
        {"index": 1, "player": "player_0", "move": "A1", "raw": "A1", "timestamp_ms": 1730000000001},
        {"index": 2, "player": "player_1", "action_text": "B1", "timestamp_ms": 1730000000002},
    ]
    replay_path = writer.write(
        scheduler_type="turn",
        result=result,
        move_log=move_log,
        arena_trace=[{"step_index": 1}],
        extra_meta={"env_impl": "gomoku_local_v1"},
    )

    replay_file = Path(replay_path)
    assert replay_file.is_absolute()
    assert replay_file.exists()
    payload = json.loads(replay_file.read_text(encoding="utf-8"))
    assert payload["schema"] == "gage_replay/v1"
    assert payload["meta"]["sample_id"] == "sample 1"
    assert payload["meta"]["scheduler_type"] == "turn"
    assert payload["recording"]["counts"]["action"] == 2
    assert payload["recording"]["counts"]["result"] == 1
    assert payload["result"]["result"] == "win"
    assert payload["arena_trace"][0]["step_index"] == 1

    events_file = replay_file.parent / "events.jsonl"
    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first_event = json.loads(lines[0])
    assert first_event["type"] == "action"
    assert first_event["actor"] == "player_0"
    assert first_event["move"] == "A1"
    result_event = json.loads(lines[-1])
    assert result_event["type"] == "result"
    assert result_event["winner"] == "player_0"


def test_replay_schema_writer_supports_frame_events_and_mode_both(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run_2"
    writer = ReplaySchemaWriter(run_dir=run_dir, sample_id="sample_2")
    result = _build_result()
    replay_path = writer.write(
        scheduler_type="tick",
        result=result,
        move_log=[{"step": 1, "player_id": "player_0", "action_text": "right"}],
        arena_trace=None,
        extra_meta={
            "env_impl": "retro_env_v1",
            "frame_events": [
                {
                    "type": "frame",
                    "stream_id": "main",
                    "encoding": "jpeg",
                    "uri": "frames/main/000001.jpg",
                }
            ],
        },
    )
    replay_file = Path(replay_path)
    payload = json.loads(replay_file.read_text(encoding="utf-8"))
    assert payload["recording"]["mode"] == "both"
    assert payload["recording"]["counts"]["frame"] == 1
    lines = (replay_file.parent / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    assert any(event.get("type") == "frame" for event in events)


def test_replay_schema_writer_accepts_legacy_trace_mapping_shape(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run_3"
    writer = ReplaySchemaWriter(run_dir=run_dir, sample_id="sample_legacy")
    result = _build_result()
    replay_path = writer.write(
        scheduler_type="turn",
        result=result,
        move_log=[],
        arena_trace={"schema": "gage.trace/v1", "steps": [{"step_index": 9}]},
    )

    replay_file = Path(replay_path)
    payload = json.loads(replay_file.read_text(encoding="utf-8"))
    assert payload["arena_trace"][0]["step_index"] == 9
    # NOTE: seq must be globally increasing across action/frame/result events.
    seqs = [int(event["seq"]) for event in events]
    assert seqs == [1, 2, 3]
