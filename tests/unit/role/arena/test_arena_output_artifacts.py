from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.adapters.arena import ArenaRoleAdapter
from gage_eval.role.arena.types import GameResult


def test_arena_output_externalizes_large_game_log(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    monkeypatch.setenv("GAGE_EVAL_GAME_LOG_INLINE_LIMIT", "1")
    monkeypatch.setenv("GAGE_EVAL_GAME_LOG_PREVIEW_LIMIT", "1")

    trace = ObservabilityTrace(run_id="arena-output-test")
    adapter = ArenaRoleAdapter(adapter_id="arena")
    result = GameResult(
        winner=None,
        result="draw",
        reason=None,
        move_count=2,
        illegal_move_count=0,
        final_board="",
        move_log=[{"index": 1}, {"index": 2}],
    )
    output = adapter._format_result(result, {"id": "sample-1"}, trace)

    assert "game_log_path" in output
    path = Path(output["game_log_path"])
    assert path.exists()

    payload = json.loads(path.read_text())
    assert payload["sample_id"] == "sample-1"
    assert len(payload["move_log"]) == 2
    assert output["game_log_total"] == 2
    assert output["game_log_truncated"] is True
    assert len(output.get("game_log_preview", [])) == 1


def test_arena_output_includes_arena_trace_when_present() -> None:
    adapter = ArenaRoleAdapter(adapter_id="arena")
    result = SimpleNamespace(
        winner=None,
        result="draw",
        reason=None,
        move_count=0,
        illegal_move_count=0,
        final_board="",
        move_log=[],
        rule_profile=None,
        win_direction=None,
        line_length=None,
        replay_path=None,
        arena_trace={"schema": "gage.trace/v1", "steps": []},
    )
    output = adapter._format_result(result, {"id": "sample-2"}, trace=None)
    assert output["arena_trace"]["schema"] == "gage.trace/v1"
