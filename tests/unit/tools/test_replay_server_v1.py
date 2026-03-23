from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def test_load_events_jsonl_streams_without_read_text(tmp_path: Path, monkeypatch) -> None:
    events_file = tmp_path / "events.jsonl"
    events_file.write_text(
        "\n".join(
            [
                json.dumps({"type": "action", "seq": 1}),
                json.dumps({"type": "result", "seq": 2}),
            ]
        ),
        encoding="utf-8",
    )

    original_read_text = Path.read_text

    def _fail_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        _ = args, kwargs
        if self == events_file:
            raise AssertionError("event loader should stream via open()")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _fail_read_text)

    events = rs._load_events_jsonl(events_file)  # noqa: SLF001

    assert [event["seq"] for event in events] == [1, 2]


def test_load_events_page_jsonl_returns_slice_and_has_more(tmp_path: Path) -> None:
    events_file = tmp_path / "events.jsonl"
    events_file.write_text(
        "\n".join(json.dumps({"type": "action", "seq": index}) for index in range(1, 5)),
        encoding="utf-8",
    )

    events, has_more = rs._load_events_page_jsonl(  # noqa: SLF001
        events_file,
        offset=1,
        limit=2,
    )

    assert [event["seq"] for event in events] == [2, 3]
    assert has_more is True


def test_is_origin_allowed_accepts_loopback_origin_by_default() -> None:
    assert rs._is_origin_allowed("http://127.0.0.1:5800", ())  # noqa: SLF001
    assert rs._is_origin_allowed("http://localhost:7860", ())  # noqa: SLF001


def test_is_origin_allowed_rejects_remote_origin_without_allowlist() -> None:
    assert not rs._is_origin_allowed("https://example.com", ())  # noqa: SLF001


def test_is_origin_allowed_accepts_explicit_allowlist_origin() -> None:
    assert rs._is_origin_allowed(  # noqa: SLF001
        "https://viewer.example.com",
        ("https://viewer.example.com",),
    )


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


def _build_replay_with_frame(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    replay_file = tmp_path / "run_3" / "replays" / "sample_3" / "replay.json"
    replay_file.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = replay_file.parent / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    (frame_dir / "frame_000001.jpg").write_bytes(b"\xff\xd8\xff")
    events_file = replay_file.parent / "events.jsonl"
    events_file.write_text(
        "\n".join(
            [
                json.dumps({"type": "action", "seq": 1}),
                json.dumps(
                    {
                        "type": "frame",
                        "seq": 2,
                        "step": 1,
                        "image": {"path": "frames/frame_000001.jpg"},
                    }
                ),
                json.dumps({"type": "result", "seq": 3}),
            ]
        ),
        encoding="utf-8",
    )
    replay_payload = {
        "schema": "gage_replay/v1",
        "recording": {"events_path": "events.jsonl"},
    }
    replay_file.write_text(json.dumps(replay_payload), encoding="utf-8")
    return replay_file, replay_payload


def test_resolve_frame_events_filters_frame_only(tmp_path: Path) -> None:
    replay_file, replay_payload = _build_replay_with_frame(tmp_path)
    frame_events, error = rs._resolve_frame_events(  # noqa: SLF001
        tmp_path,
        replay_file=replay_file,
        payload=replay_payload,
    )
    assert error is None
    assert len(frame_events) == 1
    assert frame_events[0]["type"] == "frame"
    assert frame_events[0]["seq"] == 2


def test_select_frame_event_supports_seq_and_index(tmp_path: Path) -> None:
    replay_file, replay_payload = _build_replay_with_frame(tmp_path)
    frame_events, error = rs._resolve_frame_events(  # noqa: SLF001
        tmp_path,
        replay_file=replay_file,
        payload=replay_payload,
    )
    assert error is None

    by_seq = rs._select_frame_event(frame_events, seq=2, index=None)  # noqa: SLF001
    assert by_seq is not None
    assert by_seq["seq"] == 2

    by_index = rs._select_frame_event(frame_events, seq=None, index=0)  # noqa: SLF001
    assert by_index is not None
    assert by_index["seq"] == 2

    missing = rs._select_frame_event(frame_events, seq=999, index=None)  # noqa: SLF001
    assert missing is None


def test_resolve_frame_image_path_keeps_base_dir_boundary(tmp_path: Path) -> None:
    replay_file, replay_payload = _build_replay_with_frame(tmp_path)
    frame_events, error = rs._resolve_frame_events(  # noqa: SLF001
        tmp_path,
        replay_file=replay_file,
        payload=replay_payload,
    )
    assert error is None
    frame_event = frame_events[0]

    frame_path = rs._resolve_frame_image_path(  # noqa: SLF001
        tmp_path,
        replay_file=replay_file,
        frame_event=frame_event,
    )
    assert frame_path is not None
    assert frame_path.name == "frame_000001.jpg"

    escaped = dict(frame_event)
    escaped["image"] = {"path": "../../outside.jpg"}
    escaped_path = rs._resolve_frame_image_path(  # noqa: SLF001
        tmp_path,
        replay_file=replay_file,
        frame_event=escaped,
    )
    assert escaped_path is None
