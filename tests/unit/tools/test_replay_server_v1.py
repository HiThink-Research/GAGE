from __future__ import annotations

import json
from pathlib import Path

from gage_eval.tools import replay_server as rs


def test_resolve_replay_path_prefers_v1_manifest(tmp_path: Path) -> None:
    replay_file = tmp_path / "run_1" / "replays" / "sample_1" / "replay.json"
    replay_file.parent.mkdir(parents=True, exist_ok=True)
    replay_file.write_text("{}", encoding="utf-8")

    resolved = rs._resolve_replay_path(  # noqa: SLF001
        tmp_path,
        replay_path=None,
        run_id="run_1",
        sample_id="sample 1",
    )
    assert resolved == replay_file


def test_resolve_replay_path_rejects_outside_base(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.json"
    resolved = rs._resolve_replay_path(  # noqa: SLF001
        tmp_path,
        replay_path=str(outside),
        run_id=None,
        sample_id=None,
    )
    assert resolved is None


def test_resolve_events_path_and_load_events(tmp_path: Path) -> None:
    replay_file = tmp_path / "run_2" / "replays" / "sample_2" / "replay.json"
    events_file = replay_file.parent / "events.jsonl"
    replay_file.parent.mkdir(parents=True, exist_ok=True)
    replay_file.write_text(
        json.dumps(
            {
                "schema": "gage_replay/v1",
                "recording": {"events_path": "events.jsonl"},
            }
        ),
        encoding="utf-8",
    )
    events_file.write_text(
        "\n".join(
            [
                json.dumps({"type": "action", "seq": 1}),
                json.dumps({"type": "result", "seq": 2}),
            ]
        ),
        encoding="utf-8",
    )

    payload = rs._load_json(replay_file)  # noqa: SLF001
    resolved_events = rs._resolve_events_path(  # noqa: SLF001
        tmp_path,
        replay_file=replay_file,
        payload=payload,
    )
    assert resolved_events == events_file
    events = rs._load_events_jsonl(events_file)  # noqa: SLF001
    assert len(events) == 2
    assert events[0]["type"] == "action"


def test_legacy_payload_to_events_includes_result() -> None:
    payload = {
        "moves": [{"player_id": "player_0", "action_text": "A1", "timestamp_ms": 1}],
        "winner": "player_0",
        "result": "win",
        "result_reason": "terminal",
    }
    events = rs._legacy_payload_to_events(payload)  # noqa: SLF001
    assert len(events) == 2
    assert events[0]["type"] == "action"
    assert events[1]["type"] == "result"
    assert events[1]["reason"] == "terminal"

