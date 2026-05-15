from __future__ import annotations

import pytest

from gage_eval.reporting.summary_generators.appworld import AppWorldSummaryGenerator


@pytest.mark.fast
def test_appworld_generator_outputs_v2_result() -> None:
    context = {
        "samples": [
            {
                "sample": {"metadata": {"appworld": {"subset": "train"}}},
                "judge_output": {"appworld": {"tgc": 0.5, "sgc": 1.0}},
            }
        ]
    }

    result = AppWorldSummaryGenerator().generate(context)

    assert result.legacy_payload["appworld_summary"]["overall"]["total"] == 1
    assert result.summary_sections[0]["generator_id"] == "appworld_summary"
    assert result.summary_sections[0]["section_id"] == "overview"
