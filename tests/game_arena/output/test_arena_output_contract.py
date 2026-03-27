from __future__ import annotations

from gage_eval.evaluation.sample_envelope import append_arena_contract, ensure_arena_header
from gage_eval.role.arena.core.game_session import GameSession
from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.role.arena.output.writer import ArenaOutputWriter
from gage_eval.role.adapters.arena import ArenaRoleAdapter
from gage_eval.role.arena.types import GameResult


def test_arena_output_writer_emits_contract_fields_and_bridges_to_sample() -> None:
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
            replay_path="artifacts/replays/sample-1/replay.json",
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
    assert serialized["artifacts"]["replay_ref"] == "artifacts/replays/sample-1/replay.json"

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
