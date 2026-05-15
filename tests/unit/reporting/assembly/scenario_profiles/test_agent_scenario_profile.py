from __future__ import annotations

import json

import pytest

from gage_eval.reporting.assembly.scenario_profiles.agent import AgentScenarioProfile
from gage_eval.reporting.evidence.reader import ReportEvidenceReader


@pytest.mark.io
def test_agent_profile_projects_trial_and_failure_refs(tmp_path) -> None:
    run_dir = tmp_path / "run"
    record = run_dir / "artifacts" / "task" / "sample" / "infra" / "sample_record.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"trial_results": [{"status": "failed", "failure": {"failure_code": "x"}}]}), encoding="utf-8")
    (run_dir / "samples.jsonl").write_text(
        json.dumps({"task_id": "task", "sample_id": "sample", "status": "failed", "artifact_refs": [{"path": "artifacts/task/sample/infra/sample_record.json", "name": "sample_record.json", "owner": "infra", "mime_type": "application/json", "size_bytes": 2, "sha256": "0" * 64}]}) + "\n",
        encoding="utf-8",
    )
    index = ReportEvidenceReader().build_index(run_dir)

    profile = AgentScenarioProfile().build(index)

    assert profile["profile_version"] == "gage.scenario.agent.v1"
    assert profile["trial_count"] == 1
    assert profile["representative_ref_ids"]


def test_agent_profile_counts_nested_agent_eval_trials() -> None:
    index = type(
        "Index",
        (),
        {
            "samples": [
                {
                    "model_output": {
                        "agent_eval": {
                            "trial_results": [
                                {
                                    "status": "completed",
                                    "artifact_refs": [
                                        {
                                            "path": "artifacts/task/sample/trials/trial_0001/infra/trial_result.json"
                                        }
                                    ],
                                },
                                {
                                    "status": "failed",
                                    "artifact_refs": [
                                        {
                                            "path": "artifacts/task/sample/trials/trial_0002/infra/trial_result.json"
                                        }
                                    ],
                                },
                            ]
                        }
                    }
                }
            ]
        },
    )()

    profile = AgentScenarioProfile().build(index)

    assert profile["trial_count"] == 2
    assert profile["failed_trial_count"] == 1
    assert profile["representative_ref_ids"] == [
        "evidence://artifacts/task/sample/trials/trial_0001/infra/trial_result.json",
        "evidence://artifacts/task/sample/trials/trial_0002/infra/trial_result.json",
    ]
