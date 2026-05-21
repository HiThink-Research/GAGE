from __future__ import annotations

import json

import pytest

from gage_eval.reporting.assembly.scenario_profiles import ScenarioProfileBuilder
from gage_eval.reporting.assembly.scenario_profiles.game import GameScenarioProfile
from gage_eval.reporting.contracts import EvidenceRef
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
        evidence_refs={
            "evidence://artifact/replay": EvidenceRef(
                ref_id="evidence://artifact/replay",
                path="replays/sample/replay.json",
            )
        },
    )

    profile = GameScenarioProfile().build(index)

    assert profile["profile_version"] == "gage.scenario.game.v1"
    assert profile["illegal_actions"]["games"] == 1
    assert profile["replay_refs"] == ["evidence://artifact/replay"]
    assert profile["move_count"] == 10


@pytest.mark.fast
def test_game_profile_treats_explicit_zero_illegal_move_count_as_authoritative() -> None:
    index = RunEvidenceIndex(
        run_dir="run",
        samples=[
            {
                "sample": {"metadata": {"game_arena": {"game_kit": "gomoku"}}},
                "judge_output": {"illegal_move_count": 0, "illegal_action_count": 3},
            }
        ],
    )

    profile = GameScenarioProfile().build(index)

    assert profile["illegal_actions"] == {"games": 0, "total": 0}


@pytest.mark.fast
def test_game_profile_records_ref_resolution_misses_without_emitting_path_refs() -> None:
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

    profiles, diagnostics = ScenarioProfileBuilder([GameScenarioProfile()]).build(index)

    assert profiles["game"]["replay_refs"] == []
    assert diagnostics["profile_ref_resolution_miss_count"] == 1
    assert diagnostics["warnings"][0]["code"] == "report_pack.profile_ref_resolution_miss"


@pytest.mark.fast
def test_game_profile_resolves_nested_model_output_replay_ref_and_move_count() -> None:
    index = RunEvidenceIndex(
        run_dir="run",
        samples=[
            {
                "sample_id": "sample-1",
                "sample": {"metadata": {"game_arena": {"game_kit": "tictactoe"}}},
                "model_output": {
                    "artifacts": {
                        "replay_ref": "replays/sample-1/replay.json",
                        "visual_session_ref": "replays/sample-1/arena_visual_session/v1/manifest.json",
                    },
                    "result": {"move_count": 7},
                },
                "judge_output": {},
            }
        ],
        evidence_refs={
            "evidence://artifact/canonical-replay": EvidenceRef(
                ref_id="evidence://artifact/canonical-replay",
                path="replays/sample-1/replay.json",
            ),
            "evidence://artifact/canonical-visual": EvidenceRef(
                ref_id="evidence://artifact/canonical-visual",
                path="replays/sample-1/arena_visual_session/v1/manifest.json",
            )
        },
    )

    profile = GameScenarioProfile().build(index)

    assert set(profile["replay_refs"]) == {
        "evidence://artifact/canonical-replay",
        "evidence://artifact/canonical-visual",
    }
    assert profile["move_count"] == 7


@pytest.mark.fast
def test_game_profile_uses_arena_footer_total_steps_for_move_count() -> None:
    index = RunEvidenceIndex(
        run_dir="run",
        samples=[
            {
                "sample": {
                    "_dataset_id": "gomoku",
                    "predict_result": [
                        {
                            "artifacts": {"replay_ref": "replays/sample-1/replay.json"},
                            "game_arena": {"total_steps": 12},
                        }
                    ],
                },
            }
        ],
        evidence_refs={
            "evidence://artifact/canonical-replay": EvidenceRef(
                ref_id="evidence://artifact/canonical-replay",
                path="replays/sample-1/replay.json",
            )
        },
    )

    profile = GameScenarioProfile().build(index)

    assert profile["replay_refs"] == ["evidence://artifact/canonical-replay"]
    assert profile["move_count"] == 12


@pytest.mark.io
def test_game_profile_falls_back_to_replay_manifest_stats_move_count(tmp_path) -> None:
    run_dir = tmp_path / "run"
    replay = run_dir / "replays" / "sample-1" / "replay.json"
    replay.parent.mkdir(parents=True)
    replay.write_text(json.dumps({"stats": {"move_count": 8}}), encoding="utf-8")
    index = RunEvidenceIndex(
        run_dir=run_dir,
        samples=[
            {
                "sample": {
                    "_dataset_id": "gomoku",
                    "predict_result": [
                        {
                            "artifacts": {"replay_ref": "replays/sample-1/replay.json"},
                        }
                    ],
                },
            }
        ],
        evidence_refs={
            "evidence://artifact/canonical-replay": EvidenceRef(
                ref_id="evidence://artifact/canonical-replay",
                path="replays/sample-1/replay.json",
            )
        },
    )

    profile = GameScenarioProfile().build(index)

    assert profile["replay_refs"] == ["evidence://artifact/canonical-replay"]
    assert profile["move_count"] == 8


@pytest.mark.fast
def test_game_profile_ignores_legacy_replay_path_for_replay_refs() -> None:
    index = RunEvidenceIndex(
        run_dir="run",
        samples=[
            {
                "sample": {"_dataset_id": "gomoku"},
                "replay_path": "replays/legacy/replay.json",
            }
        ],
        evidence_refs={
            "evidence://artifact/legacy-replay": EvidenceRef(
                ref_id="evidence://artifact/legacy-replay",
                path="replays/legacy/replay.json",
            )
        },
    )

    profile = GameScenarioProfile().build(index)

    assert profile["replay_refs"] == []
    assert profile["move_count"] == 0
