from __future__ import annotations

import json

import pytest

from gage_eval.reporting.assembly.scenario_profiles.game import GameScenarioProfile
from gage_eval.reporting.evidence.reader import ReportEvidenceReader, RunEvidenceIndex


@pytest.mark.io
def test_game_profile_integration() -> None:
    index = RunEvidenceIndex(run_dir="run", samples=[{"sample": {"_dataset_id": "gomoku"}, "judge_output": {}}])

    profile = GameScenarioProfile().build(index)

    assert "gomoku" in profile["game_kits"]


@pytest.mark.io
def test_game_profile_integrates_reader_replay_evidence(tmp_path) -> None:
    run_dir = tmp_path / "run"
    replay = run_dir / "replays" / "gomoku_match_0001" / "replay.json"
    visual = replay.parent / "arena_visual_session" / "v1" / "manifest.json"
    visual.parent.mkdir(parents=True)
    replay.write_text('{"stats":{"move_count":9}}', encoding="utf-8")
    visual.write_text('{"version":"v1"}', encoding="utf-8")
    (run_dir / "samples.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "gomoku_match_0001",
                "sample": {"_dataset_id": "gomoku"},
                "model_output": {
                    "artifacts": {"replay_ref": str(replay)},
                    "result": {
                        "artifacts": {"visual_session_ref": str(visual)},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    index = ReportEvidenceReader().build_index(run_dir)
    profile = GameScenarioProfile().build(index)

    assert len(index.evidence_refs) == 2
    assert set(profile["replay_refs"]) == set(index.evidence_refs)
    assert profile["move_count"] == 9


@pytest.mark.io
@pytest.mark.parametrize(
    ("sample_payload", "expected_paths"),
    [
        (
            {"artifacts": {"replay_ref": "replays/sample/replay.json"}},
            {"replays/sample/replay.json"},
        ),
        (
            {"model_output": {"result": {"artifacts": {"replay_v1_ref": "replays/sample/replay-v1.json"}}}},
            {"replays/sample/replay-v1.json"},
        ),
        (
            {
                "sample": {
                    "predict_result": [
                        {
                            "result": {
                                "artifacts": {
                                    "visual_session_ref": "replays/sample/arena_visual_session/v1/manifest.json"
                                }
                            }
                        }
                    ]
                }
            },
            {"replays/sample/arena_visual_session/v1/manifest.json"},
        ),
    ],
)
def test_game_profile_and_reader_discover_game_artifacts_from_same_sources(
    tmp_path,
    sample_payload,
    expected_paths,
) -> None:
    run_dir = tmp_path / "run"
    for ref_path in expected_paths:
        artifact = run_dir / ref_path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}", encoding="utf-8")
    sample = {"sample_id": "sample", "sample": {"_dataset_id": "gomoku"}}
    sample.update(sample_payload)
    (run_dir / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")

    index = ReportEvidenceReader().build_index(run_dir)
    profile = GameScenarioProfile().build(index)

    assert {ref.path for ref in index.evidence_refs.values()} == expected_paths
    assert set(profile["replay_refs"]) == set(index.evidence_refs)
