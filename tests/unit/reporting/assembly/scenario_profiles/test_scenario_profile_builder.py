from __future__ import annotations

import pytest

from gage_eval.reporting.assembly.scenario_profiles import ScenarioProfileBuilder
from gage_eval.reporting.assembly.scenario_profiles.game import GameScenarioProfile
from gage_eval.reporting.evidence.reader import RunEvidenceIndex


class _FailingProfile:
    profile_name = "broken"

    def build(self, index):
        raise RuntimeError("boom")


@pytest.mark.fast
def test_scenario_profile_failure_is_degraded() -> None:
    profiles, diagnostics = ScenarioProfileBuilder([_FailingProfile()]).build(index=object())

    assert profiles == {}
    assert diagnostics["warnings"][0]["code"] == "report_pack.scenario_profile_failed"


@pytest.mark.fast
def test_scenario_profile_builder_does_not_mutate_index_diagnostics_miss_count() -> None:
    index = RunEvidenceIndex(
        run_dir="run",
        samples=[
            {
                "sample": {"metadata": {"game_arena": {"game_kit": "tictactoe"}}},
                "artifact_refs": [{"name": "replay.json", "path": "replays/sample/replay.json"}],
            }
        ],
        evidence_refs={},
    )

    first_profiles, first_diagnostics = ScenarioProfileBuilder([GameScenarioProfile()]).build(index)
    second_profiles, second_diagnostics = ScenarioProfileBuilder([GameScenarioProfile()]).build(index)

    assert first_profiles["game"]["replay_refs"] == []
    assert second_profiles["game"]["replay_refs"] == []
    assert first_diagnostics["profile_ref_resolution_miss_count"] == 1
    assert second_diagnostics["profile_ref_resolution_miss_count"] == 1
    assert index.diagnostics.profile_ref_resolution_miss_count == 0
