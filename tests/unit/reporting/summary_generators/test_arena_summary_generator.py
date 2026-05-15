from __future__ import annotations

import pytest

from gage_eval.reporting.summary_generators.arena import ArenaSummaryGenerator


@pytest.mark.fast
def test_arena_generator_outputs_game_section() -> None:
    context = {
        "samples": [
            {
                "sample": {
                    "metadata": {"game_arena": {"start_time_ms": 0}},
                    "predict_result": [{"game_arena": {"end_time_ms": 10, "total_steps": 3, "winner_player_id": "p1"}}],
                }
            }
        ]
    }

    result = ArenaSummaryGenerator().generate(context)

    assert result.legacy_payload["arena_summary"]["overall"]["samples"] == 1
    assert result.summary_sections[0]["generator_id"] == "arena_summary"
    assert result.summary_sections[0]["section_id"] == "overview"
    assert result.outliers
