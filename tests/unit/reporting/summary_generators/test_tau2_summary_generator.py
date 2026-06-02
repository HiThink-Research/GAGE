from __future__ import annotations

import pytest

from gage_eval.reporting.summary_generators.tau2 import Tau2SummaryGenerator


@pytest.mark.fast
def test_tau2_generator_outputs_v2_result_and_legacy_summary() -> None:
    context = {
        "samples": [
            {
                "sample": {"id": "s1", "metadata": {"tau2": {"task_id": "1", "domain": "airline"}}},
                "judge_output": {"tau2": {"reward": 1.0, "agent_cost": 2.0, "user_cost": 1.0}},
            }
        ],
        "evidence_refs": [],
    }

    result = Tau2SummaryGenerator().generate(context)

    assert result.generator_id == "tau2_summary"
    assert result.legacy_payload["tau2_summary"]["overall"]["avg_reward"] == 1.0
    assert result.summary_sections
    assert result.summary_sections[0]["section_id"] == "overview"
    assert result.summary_sections[0]["metrics"]["section_id"] == "overview"
