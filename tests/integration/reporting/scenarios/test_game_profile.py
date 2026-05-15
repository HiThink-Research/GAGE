from __future__ import annotations

import pytest

from gage_eval.reporting.assembly.scenario_profiles.game import GameScenarioProfile
from gage_eval.reporting.evidence.reader import RunEvidenceIndex


@pytest.mark.io
def test_game_profile_integration() -> None:
    index = RunEvidenceIndex(run_dir="run", samples=[{"sample": {"_dataset_id": "gomoku"}, "judge_output": {}}])

    profile = GameScenarioProfile().build(index)

    assert "gomoku" in profile["game_kits"]
