from __future__ import annotations

from types import SimpleNamespace

from gage_eval.evaluation.sample_envelope import append_arena_contract, ensure_arena_header
from gage_eval.role.arena.core.game_session import GameSession
from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.role.arena.output.writer import ArenaOutputWriter
from gage_eval.role.arena.visualization.recorder import ArenaVisualSessionRecorder
from gage_eval.role.adapters.arena import ArenaRoleAdapter
from gage_eval.role.arena.types import GameResult


def test_arena_output_writer_emits_contract_fields_and_bridges_to_sample() -> None:
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena-role",
        game_id="gomoku",
        scheduling_family="turn",
        session_id="sample-1",
    )
    replay_path = "artifacts/replays/sample-1/replay.json"
    recorder.record_decision_window_open(
        ts_ms=1001,
        step=0,
        tick=0,
        player_id="Black",
        observation={"board_text": "board"},
    )
    recorder.record_action_intent(
        ts_ms=1002,
        step=0,
        tick=0,
        player_id="Black",
        action={"move": "A1"},
    )
    recorder.record_action_committed(
        ts_ms=1003,
        step=0,
        tick=0,
        player_id="Black",
        action={"move": "A1"},
    )
    recorder.record_decision_window_close(
        ts_ms=1004,
        step=1,
        tick=1,
        player_id="White",
    )
    recorder.record_snapshot(
        ts_ms=1005,
        step=1,
        tick=1,
        snapshot={"board_text": "board"},
    )
    recorder.persist(replay_path)

    session = GameSession(
        sample=ArenaSample(
            game_kit="gomoku",
            env="gomoku_standard",
            scheduler="turn/default",
            players=(
                {"player_id": "Black", "seat": "black"},
                {"player_id": "White", "seat": "white"},
            ),
            runtime_overrides={"board_size": 9},
        ),
        tick=3,
        step=3,
        final_result=GameResult(
            winner="Black",
            result="win",
            reason="five_in_row",
            move_count=3,
            illegal_move_count=0,
            final_board="board",
            move_log=[{"index": 1, "player": "Black", "move": "A1"}],
            replay_path=replay_path,
        ),
        arena_trace=[
            {
                "step_index": 0,
                "trace_state": "done",
                "timestamp": 1001,
                "player_id": "Black",
                "action_raw": "A1",
                "action_applied": "A1",
                "t_obs_ready_ms": 1001,
                "t_action_submitted_ms": 1003,
                "timeout": False,
                "is_action_legal": True,
                "retry_count": 0,
            }
        ],
        support_workflow=SimpleNamespace(
            workflow_id="arena/default",
            metadata={"workflow_kind": "game_support"},
        ),
        visualization_spec=SimpleNamespace(
            spec_id="arena/visualization/gomoku_board_v1",
            visual_kind="board",
        ),
        resources=SimpleNamespace(
            resource_categories=("game_runtime_resource", "visualization_resource"),
            lifecycle_phase="allocated",
            resource_artifacts={"frame_capture_ref": "artifacts/frames/sample-1/frame-0.png"},
        ),
        visual_recorder=recorder,
    )

    output = ArenaOutputWriter().finalize(session)
    serialized = ArenaRoleAdapter._serialize_gamearena_value(output)

    assert serialized["sample"]["game_kit"] == "gomoku"
    assert serialized["output_kind"] == "arena"
    assert serialized["tick"] == 3
    assert serialized["step"] == 3
    assert serialized["arena_trace"][0]["player_id"] == "Black"
    assert serialized["header"]["scheduler"] == "turn/default"
    assert serialized["trace"][0]["step_index"] == 0
    assert serialized["footer"]["winner_player_id"] == "Black"
    assert serialized["footer"]["termination_reason"] == "five_in_row"
    assert serialized["footer"]["total_steps"] == 3
    assert serialized["resource_artifacts"]["frame_capture_ref"] == (
        "artifacts/frames/sample-1/frame-0.png"
    )
    assert serialized["resource_artifacts"]["replay_ref"] == "artifacts/replays/sample-1/replay.json"
    assert serialized["game_context"]["game_kit"] == "gomoku"
    assert serialized["game_context"]["env"] == "gomoku_standard"
    assert serialized["game_context"]["support_workflow"] == "arena/default"
    assert serialized["game_context"]["visualization_spec"] == "arena/visualization/gomoku_board_v1"
    assert serialized["game_context"]["resource_categories"] == (
        "game_runtime_resource",
        "visualization_resource",
    )
    assert serialized["artifacts"]["replay_ref"] == "artifacts/replays/sample-1/replay.json"
    assert serialized["artifacts"]["visual_session_ref"].endswith(
        "arena_visual_session/v1/manifest.json"
    )

    sample = {
        "id": "sample-1",
        "metadata": {"player_ids": ["Black", "White"]},
        "predict_result": [],
    }
    ensure_arena_header(sample, start_time_ms=1000)
    append_arena_contract(sample, serialized, end_time_ms=2000)

    entry = sample["predict_result"][0]
    assert sample["metadata"]["game_arena"]["game_kit"] == "gomoku"
    assert sample["metadata"]["game_arena"]["env"] == "gomoku_standard"
    assert sample["metadata"]["game_arena"]["scheduler"] == "turn/default"
    assert entry["trace"][0]["player_id"] == "Black"
    assert entry["arena_trace"][0]["player_id"] == "Black"
    assert entry["game_arena"]["winner_player_id"] == "Black"
    assert entry["game_arena"]["termination_reason"] == "five_in_row"
    assert entry["game_arena"]["total_steps"] == 3
    assert entry["artifacts"]["replay_ref"] == "artifacts/replays/sample-1/replay.json"


def test_arena_output_writer_serializes_support_and_resource_failures() -> None:
    session = GameSession(
        sample=ArenaSample(game_kit="gomoku", env="gomoku_standard"),
        final_result=GameResult(
            winner="Black",
            result="win",
            reason="completed",
            move_count=1,
            illegal_move_count=0,
            final_board="board",
            move_log=[{"index": 1, "player": "Black", "move": "A1"}],
        ),
        support_errors=[
            {
                "error_code": "support_workflow_failure",
                "workflow_id": "arena/default",
                "hook": "before_apply",
                "unit_id": "test/unit",
                "unit_kind": "execution_support",
                "message": "support unit failed",
            }
        ],
        resources=SimpleNamespace(
            resource_categories=("game_runtime_resource",),
            lifecycle_phase="release_failed",
            errors=[
                {
                    "error_code": "resource_lifecycle_error",
                    "resource_category": "game_runtime_resource",
                    "operation": "close",
                    "message": "close failed",
                }
            ],
        ),
    )

    output = ArenaOutputWriter().finalize(session)
    serialized = ArenaRoleAdapter._serialize_gamearena_value(output)

    assert serialized["game_context"]["support_errors"][0]["error_code"] == (
        "support_workflow_failure"
    )
    assert serialized["game_context"]["support_errors"][0]["workflow_id"] == "arena/default"
    assert serialized["game_context"]["resource_errors"][0]["error_code"] == (
        "resource_lifecycle_error"
    )
    assert serialized["game_context"]["resource_errors"][0]["operation"] == "close"
