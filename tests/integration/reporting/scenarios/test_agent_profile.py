from __future__ import annotations

import pytest

from gage_eval.reporting.assembly.scenario_profiles.agent import AgentScenarioProfile
from gage_eval.reporting.evidence.reader import RunEvidenceIndex


@pytest.mark.io
def test_agent_profile_integration_from_sample_records() -> None:
    index = RunEvidenceIndex(run_dir="run", samples=[{"trial_results": [{"status": "completed"}]}])

    profile = AgentScenarioProfile().build(index)

    assert profile["trial_count"] == 1
