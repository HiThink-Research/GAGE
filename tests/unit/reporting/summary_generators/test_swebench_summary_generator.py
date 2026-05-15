from __future__ import annotations

import pytest

from gage_eval.reporting.summary_generators.swebench import SwebenchSummaryGenerator


@pytest.mark.fast
def test_swebench_generator_outputs_unresolved_attention() -> None:
    context = {
        "samples": [
            {
                "sample": {"_dataset_id": "swebench", "metadata": {"repo": "owner/repo"}},
                "judge_output": {"resolved": False, "failure_reason": "tests_failed"},
            }
        ]
    }

    result = SwebenchSummaryGenerator().generate(context)

    assert result.legacy_payload["swebench_summary"]["overall"]["resolve_rate"] == 0.0
    assert result.summary_sections[0]["generator_id"] == "swebench_summary"
    assert result.summary_sections[0]["section_id"] == "overview"
    assert result.attention_cases[0]["reason_codes"] == ["score.low"]
