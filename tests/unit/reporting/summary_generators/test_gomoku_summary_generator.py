from __future__ import annotations

import pytest

from gage_eval.reporting.summary_generators.gomoku import GomokuSummaryGenerator


@pytest.mark.fast
def test_gomoku_generator_outputs_illegal_action_case() -> None:
    context = {
        "samples": [
            {
                "sample": {"_dataset_id": "gomoku", "metadata": {"player_ids": ["p1", "p2"]}},
                "judge_output": {"winner": "p1", "illegal_move_count": 1, "move_count": 3},
            }
        ]
    }

    result = GomokuSummaryGenerator().generate(context)

    assert result.legacy_payload["gomoku_summary"]["overall"]["illegal_games"] == 1
    assert result.summary_sections[0]["generator_id"] == "gomoku_summary"
    assert result.summary_sections[0]["section_id"] == "overview"
    assert result.attention_cases[0]["reason_codes"] == ["game.illegal_action"]


@pytest.mark.fast
def test_gomoku_generator_reads_nested_arena_result_payload() -> None:
    context = {
        "samples": [
            {
                "sample": {"_dataset_id": "gomoku", "metadata": {"player_ids": ["Black", "White"]}},
                "model_output": {
                    "result": {
                        "winner": None,
                        "result": "draw",
                        "move_count": 9,
                        "illegal_move_count": 0,
                    }
                },
            }
        ]
    }

    result = GomokuSummaryGenerator().generate(context)
    summary = result.legacy_payload["gomoku_summary"]

    assert summary["overall"]["draws"] == 1
    assert summary["overall"]["avg_moves"] == 9.0
    assert summary["results"] == {"draw": 1}
