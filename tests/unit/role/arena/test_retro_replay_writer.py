import json
from pathlib import Path

import pytest

from gage_eval.role.arena.games.retro.replay import ReplaySchemaWriter
from gage_eval.role.arena.types import ArenaAction, GameResult


def test_replay_writer_append_decision_records_metadata_and_hold_ticks():
    writer = ReplaySchemaWriter(
        game="SuperMarioBros3-Nes-v0",
        state="Start",
        run_id=None,
        sample_id="sample_0",
        replay_output_dir=None,
        replay_filename=None,
        frame_output_dir=None,
        frame_stride=2,
        snapshot_stride=2,
        rom_path=None,
    )

    action = ArenaAction(
        player="player_0",
        move="right",
        raw="right",
        metadata={
            "hold_ticks": 9,
            "error": "none",
            "latency_ms": 12,
            "timed_out": False,
            "fallback_used": None,
            "llm_wait_mode": "block",
        },
    )
    writer.append_decision(action, decision_index=0, start_tick=1, end_tick=3)

    assert len(writer._moves) == 1  # type: ignore[attr-defined]
    assert writer._moves[0]["hold_ticks"] == 9  # type: ignore[attr-defined]
    assert writer._moves[0]["latency_ms"] == 12  # type: ignore[attr-defined]


def test_replay_writer_append_tick_respects_snapshot_and_frame_stride():
    np = pytest.importorskip("numpy")

    writer = ReplaySchemaWriter(
        game="SuperMarioBros3-Nes-v0",
        state="Start",
        run_id=None,
        sample_id="sample_0",
        replay_output_dir=None,
        replay_filename=None,
        frame_output_dir=None,
        frame_stride=2,
        snapshot_stride=3,
        rom_path=None,
    )

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    for tick in range(1, 7):
        writer.append_tick(tick=tick, reward=1.0, info={"tick": tick}, frame=frame, done=False)

    assert [item["tick"] for item in writer._snapshots] == [3, 6]  # type: ignore[attr-defined]
    assert [item["tick"] for item in writer._frames] == [2, 4, 6]  # type: ignore[attr-defined]
    assert writer._frames[0]["frame_path"] is None  # type: ignore[attr-defined]
    assert writer._frames[0]["shape"] == (2, 2, 3)  # type: ignore[attr-defined]


@pytest.mark.io
def test_replay_writer_finalize_writes_replay_json(tmp_path):
    writer = ReplaySchemaWriter(
        game="SuperMarioBros3-Nes-v0",
        state="Start",
        run_id=None,
        sample_id="sample 1",
        replay_output_dir=str(tmp_path),
        replay_filename=None,
        frame_output_dir=None,
        frame_stride=1,
        snapshot_stride=1,
        rom_path="/roms/smb3.nes",
    )

    result = GameResult(
        winner=None,
        result="draw",
        reason=None,
        move_count=0,
        illegal_move_count=0,
        final_board="{}",
        move_log=[],
    )

    replay_path = writer.finalize(result)
    assert replay_path is not None

    replay_file = tmp_path / "sample_1" / "replay.json"
    payload = json.loads(replay_file.read_text(encoding="utf-8"))
    assert payload["meta"]["rom_path"] == "/roms/smb3.nes"
    assert payload["result"]["status"] == "draw"


def test_replay_writer_finalize_returns_none_when_write_fails(monkeypatch: pytest.MonkeyPatch, tmp_path):
    writer = ReplaySchemaWriter(
        game="SuperMarioBros3-Nes-v0",
        state="Start",
        run_id=None,
        sample_id="sample_1",
        replay_output_dir=str(tmp_path),
        replay_filename=None,
        frame_output_dir=None,
        frame_stride=1,
        snapshot_stride=1,
        rom_path=None,
    )
    result = GameResult(
        winner=None,
        result="draw",
        reason=None,
        move_count=0,
        illegal_move_count=0,
        final_board="{}",
        move_log=[],
    )

    monkeypatch.setattr(Path, "write_text", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("boom")))
    assert writer.finalize(result) is None


@pytest.mark.io
def test_replay_writer_append_tick_writes_frame_file_when_output_dir_set(tmp_path):
    np = pytest.importorskip("numpy")
    frame_dir = tmp_path / "frames"
    writer = ReplaySchemaWriter(
        game="SuperMarioBros3-Nes-v0",
        state="Start",
        run_id=None,
        sample_id="sample_0",
        replay_output_dir=str(tmp_path),
        replay_filename="replay.json",
        frame_output_dir=str(frame_dir),
        frame_stride=1,
        snapshot_stride=2,
        rom_path=None,
    )

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    writer.append_tick(tick=1, reward=0.0, info={"tick": 1}, frame=frame, done=False)

    assert len(writer._frames) == 1  # type: ignore[attr-defined]
    frame_path = writer._frames[0]["frame_path"]  # type: ignore[attr-defined]
    assert isinstance(frame_path, str)
    assert Path(frame_path).exists()


def test_replay_writer_resolve_output_path_uses_env_run_id(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    monkeypatch.setenv("GAGE_EVAL_RUN_ID", "run_env")
    monkeypatch.setenv("GAGE_EVAL_SAMPLE_ID", "sample env")

    writer = ReplaySchemaWriter(
        game="SuperMarioBros3-Nes-v0",
        state="Start",
        run_id=None,
        sample_id=None,
        replay_output_dir=None,
        replay_filename=None,
        frame_output_dir=None,
        frame_stride=1,
        snapshot_stride=1,
        rom_path=None,
    )

    resolved = writer._resolve_replay_output_path()  # noqa: SLF001
    assert resolved == tmp_path / "run_env" / "replays" / "sample_env" / "replay.json"
