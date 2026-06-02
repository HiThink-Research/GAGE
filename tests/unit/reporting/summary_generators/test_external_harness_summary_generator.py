from __future__ import annotations

import pytest

from gage_eval.reporting.summary_generators.external_harness import ExternalHarnessSummaryGenerator


@pytest.mark.fast
def test_external_harness_generator_outputs_failure_rollup_and_refs() -> None:
    context = {
        "samples": [
            {
                "sample": {"task_type": "external_harness.harbor", "dataset_id": "tb2"},
                "trial_results": [{"trial_id": "t1", "status": "aborted", "failure": {"failure_code": "x"}}],
                "artifact_refs": [{"path": "artifacts/task/sample/infra/harbor_raw_result.json"}],
            }
        ]
    }

    result = ExternalHarnessSummaryGenerator().generate(context)

    assert result.legacy_payload["external_harness"]["sample_count"] == 1
    assert result.summary_sections[0]["generator_id"] == "external_harness_summary"
    assert result.summary_sections[0]["section_id"] == "overview"
    assert result.attention_cases
