from __future__ import annotations

import json

import pytest

from gage_eval.reporting.assembly.scenario_profiles.agent import AgentScenarioProfile
from gage_eval.reporting.contracts import EvidenceRef
from gage_eval.reporting.evidence.reader import ReportEvidenceReader
from gage_eval.reporting.evidence.reader import RunEvidenceIndex


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
    assert profile["representative_ref_ids"][0].startswith("evidence://artifact/")


def test_agent_profile_counts_nested_agent_eval_trials() -> None:
    first_path = "artifacts/task/sample/trials/trial_0001/infra/trial_result.json"
    second_path = "artifacts/task/sample/trials/trial_0002/infra/trial_result.json"
    index = type(
        "Index",
        (),
        {
            "evidence_refs": {
                "evidence://artifact/first": EvidenceRef(ref_id="evidence://artifact/first", path=first_path),
                "evidence://artifact/second": EvidenceRef(ref_id="evidence://artifact/second", path=second_path),
            },
            "samples": [
                {
                    "model_output": {
                        "agent_eval": {
                            "trial_results": [
                                {
                                    "status": "completed",
                                    "artifact_refs": [
                                        {
                                            "path": first_path
                                        }
                                    ],
                                },
                                {
                                    "status": "failed",
                                    "artifact_refs": [
                                        {
                                            "path": second_path
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
        "evidence://artifact/first",
        "evidence://artifact/second",
    ]


def test_agent_profile_omits_missing_canonical_artifact_refs() -> None:
    index = RunEvidenceIndex(
        run_dir="run",
        samples=[
            {
                "artifact_refs": [
                    {
                        "path": "artifacts/task/sample/infra/sample_record.json",
                        "name": "sample_record.json",
                    }
                ]
            }
        ],
        evidence_refs={},
    )

    profile = AgentScenarioProfile().build(index)

    assert profile["representative_ref_ids"] == []
