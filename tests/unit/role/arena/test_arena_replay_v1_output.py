from __future__ import annotations

import json
from pathlib import Path

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.adapters.arena import ArenaRoleAdapter
from gage_eval.role.arena.types import GameResult


def _build_result(*, replay_path: str | None = None) -> GameResult:
    return GameResult(
        winner="player_0",
        result="win",
        reason="terminal",
        move_count=1,
        illegal_move_count=0,
        final_board="{}",
        move_log=[{"index": 1, "player": "player_0", "move": "A1", "raw": "A1"}],
        replay_path=replay_path,
    )


def test_arena_result_dual_writes_replay_v1_when_primary_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    trace = ObservabilityTrace(run_id="run_dual")
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={
            "impl": "gomoku_local_v1",
            "replay": {"enabled": True, "primary_mode": False},
        },
        scheduler={"type": "turn"},
    )
    result = _build_result(replay_path="legacy_replay.json")
    output = adapter._format_result(result, {"id": "sample 1"}, trace)

    assert output["replay_path"] == "legacy_replay.json"
    assert "replay_v1_path" in output
    replay_file = Path(output["replay_v1_path"])
    assert replay_file.exists()
    payload = json.loads(replay_file.read_text(encoding="utf-8"))
    assert payload["schema"] == "gage_replay/v1"
    assert payload["meta"]["run_id"] == "run_dual"


def test_arena_result_primary_mode_replaces_replay_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    trace = ObservabilityTrace(run_id="run_primary")
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={
            "impl": "gomoku_local_v1",
            "replay": {"enabled": True, "primary_mode": True},
        },
        scheduler={"type": "turn"},
    )
    result = _build_result(replay_path="legacy_replay.json")
    output = adapter._format_result(result, {"id": "sample-2"}, trace)

    assert "replay_v1_path" not in output
    replay_path = output["replay_path"]
    assert replay_path.endswith("replay.json")
    replay_file = Path(replay_path)
    assert replay_file.exists()


def test_arena_result_replay_v1_disabled_keeps_legacy_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    trace = ObservabilityTrace(run_id="run_legacy")
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "gomoku_local_v1", "replay": {"enabled": False}},
        scheduler={"type": "turn"},
    )
    result = _build_result(replay_path="legacy_replay.json")
    output = adapter._format_result(result, {"id": "sample-3"}, trace)

    assert output["replay_path"] == "legacy_replay.json"
    assert "replay_v1_path" not in output
