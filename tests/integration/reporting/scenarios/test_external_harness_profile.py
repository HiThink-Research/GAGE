from __future__ import annotations

import pytest

from gage_eval.reporting.assembly.scenario_profiles.external_harness import ExternalHarnessScenarioProfile
from gage_eval.reporting.evidence.reader import RunEvidenceIndex


@pytest.mark.io
def test_external_harness_profile_integration() -> None:
    index = RunEvidenceIndex(run_dir="run", samples=[{"sample": {"task_type": "external_harness.harbor"}, "trial_results": []}])

    profile = ExternalHarnessScenarioProfile().build(index)

    assert profile["harnesses"][0]["harness_id"] == "harbor"
