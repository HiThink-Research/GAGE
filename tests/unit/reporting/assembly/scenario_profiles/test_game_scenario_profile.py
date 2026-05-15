from __future__ import annotations

import pytest

from gage_eval.reporting.assembly.scenario_profiles.game import GameScenarioProfile
from gage_eval.reporting.evidence.reader import RunEvidenceIndex


@pytest.mark.fast
def test_game_profile_projects_illegal_actions_and_replays() -> None:
    index = RunEvidenceIndex(
        run_dir="run",
        samples=[
            {
                "sample": {"metadata": {"game_arena": {"game_kit": "gomoku"}}},
                "judge_output": {"winner": "p1", "illegal_move_count": 1, "move_count": 10},
                "artifact_refs": [{"name": "replay.json", "path": "replays/sample/replay.json"}],
            }
        ],
    )

    profile = GameScenarioProfile().build(index)

    assert profile["profile_version"] == "gage.scenario.game.v1"
    assert profile["illegal_actions"]["games"] == 1
    assert profile["replay_refs"]
