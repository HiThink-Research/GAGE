from __future__ import annotations

import json

import pytest

from gage_eval.reporting.evidence.reader import ReportEvidenceReader


@pytest.mark.io
def test_reader_builds_index_from_minimal_run(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"sample_count": 1}), encoding="utf-8")
    (run_dir / "samples.jsonl").write_text(json.dumps({"sample_id": "sample-1", "status": "completed"}) + "\n", encoding="utf-8")

    index = ReportEvidenceReader().build_index(run_dir)

    assert index.summary["sample_count"] == 1
    assert len(index.samples) == 1
    assert all(not ref.path.startswith("/") for ref in index.evidence_refs.values())


@pytest.mark.io
def test_reader_redacts_secret_preview_and_records_diagnostic(tmp_path) -> None:
    run_dir = tmp_path / "run"
    artifact = run_dir / "artifacts" / "task" / "sample" / "infra" / "raw_error.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"error": "Authorization: Bearer abc123"}), encoding="utf-8")
    (run_dir / "samples.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "task_id": "task",
                "artifact_refs": [
                    {
                        "owner": "infra",
                        "name": "raw_error.json",
                        "path": "artifacts/task/sample/infra/raw_error.json",
                        "mime_type": "application/json",
                        "size_bytes": artifact.stat().st_size,
                        "sha256": "0" * 64,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    index = ReportEvidenceReader().build_index(run_dir)

    previews = [ref.preview or {} for ref in index.evidence_refs.values()]
    assert any("<redacted:" in str(preview) for preview in previews)
    assert any(item["code"] == "report_pack.secret_leak_detected" for item in index.diagnostics.warnings)


@pytest.mark.io
def test_reader_indexes_agentkit_nested_trial_artifacts_and_skips_missing_refs(tmp_path) -> None:
    run_dir = tmp_path / "run"
    artifact = run_dir / "artifacts" / "task" / "sample" / "trials" / "trial_0001" / "infra" / "trial_result.json"
    trace = artifact.parent / "trace.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    trace.write_text(json.dumps({"event_type": "trial.end"}) + "\n", encoding="utf-8")
    (run_dir / "samples.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "task_id": "task",
                "judge_output": {
                    "artifact_refs": [
                        {"owner": "verifier", "name": "missing.json", "path": "verifier/result.json"}
                    ]
                },
                "model_output": {
                    "agent_eval": {
                        "trial_results": [
                            {
                                "artifact_refs": [
                                    {
                                        "owner": "infra",
                                        "name": "trial_result.json",
                                        "path": "artifacts/task/sample/trials/trial_0001/infra/trial_result.json",
                                    }
                                ],
                                "trace_ref": {
                                    "owner": "infra",
                                    "name": "trace.jsonl",
                                    "path": "artifacts/task/sample/trials/trial_0001/infra/trace.jsonl",
                                },
                            }
                        ]
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    index = ReportEvidenceReader().build_index(run_dir)

    paths = {ref.path for ref in index.evidence_refs.values()}
    assert "artifacts/task/sample/trials/trial_0001/infra/trial_result.json" in paths
    assert "artifacts/task/sample/trials/trial_0001/infra/trace.jsonl" in paths
    assert "verifier/result.json" not in paths
    assert all(ref.timestamp_iso and ref.size_bytes is not None and ref.sha256 for ref in index.evidence_refs.values())
    assert any(item["code"] == "report_pack.artifact_missing" for item in index.diagnostics.warnings)
