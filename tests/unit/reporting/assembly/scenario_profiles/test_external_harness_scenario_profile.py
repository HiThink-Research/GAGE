from __future__ import annotations

import pytest

from gage_eval.reporting.assembly.scenario_profiles.external_harness import ExternalHarnessScenarioProfile
from gage_eval.reporting.evidence.reader import RunEvidenceIndex


@pytest.mark.fast
def test_external_harness_profile_uses_generic_contract_fields() -> None:
    index = RunEvidenceIndex(run_dir="run", samples=[{"sample": {"task_type": "external_harness.harbor"}, "trial_results": [{"trial_id": "t1", "status": "aborted"}]}])

    profile = ExternalHarnessScenarioProfile().build(index)

    assert profile["profile_version"] == "gage.scenario.external_harness.v1"
    assert profile["trial_rollup"]["aborted"] == 1
    assert "harbor_task_key" not in str(profile)
