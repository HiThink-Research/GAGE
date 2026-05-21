from __future__ import annotations

import hashlib
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
def test_reader_indexes_static_sample_records_as_evidence(tmp_path) -> None:
    run_dir = tmp_path / "run"
    sample_record = run_dir / "samples" / "task_static" / "task_static_sample-1.json"
    sample_record.parent.mkdir(parents=True)
    record = {
        "sample_id": "task_static:sample-1",
        "task_id": "task_static",
        "namespace": "task_static",
        "model_output": {"answer": "A"},
        "judge_output": {"score": 1.0},
        "metrics": {"acc": {"values": {"acc": 1.0}}},
    }
    sample_record.write_text(json.dumps(record), encoding="utf-8")
    (run_dir / "samples.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    index = ReportEvidenceReader().build_index(run_dir)

    refs = list(index.evidence_refs.values())
    assert len(refs) == 1
    ref = refs[0]
    assert ref.kind == "artifact"
    assert ref.artifact_role == "sample_record"
    assert ref.path == "samples/task_static/task_static_sample-1.json"
    assert ref.sample_id == "task_static:sample-1"
    assert ref.task_id == "task_static"


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
def test_reader_skips_relative_artifact_symlink_that_escapes_run_dir(tmp_path) -> None:
    run_dir = tmp_path / "run"
    outside_dir = tmp_path / "outside"
    artifact = run_dir / "artifacts" / "escape.txt"
    outside_file = outside_dir / "escape.txt"
    artifact.parent.mkdir(parents=True)
    outside_dir.mkdir()
    outside_file.write_text("outside", encoding="utf-8")
    try:
        artifact.symlink_to(outside_file)
    except OSError as exc:
        pytest.skip(f"symlink creation is not available in this environment: {exc}")
    (run_dir / "samples.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "task_id": "task",
                "artifact_refs": [
                    {
                        "owner": "infra",
                        "name": "escape.txt",
                        "path": "artifacts/escape.txt",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    index = ReportEvidenceReader().build_index(run_dir)

    assert index.evidence_refs == {}
    assert any(item["code"] == "report_pack.artifact_ref_path_escapes_root" for item in index.diagnostics.warnings)


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


@pytest.mark.io
def test_reader_indexes_arena_replay_and_visual_session_refs_from_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "run"
    replay = run_dir / "replays" / "sample-1" / "replay.json"
    visual = replay.parent / "arena_visual_session" / "v1" / "manifest.json"
    visual.parent.mkdir(parents=True)
    replay.write_text(json.dumps({"moves": [{"step": 1}]}), encoding="utf-8")
    visual.write_text(json.dumps({"version": "v1"}), encoding="utf-8")
    (run_dir / "samples.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "task_id": "gomoku",
                "model_output": {
                    "artifacts": {
                        "replay_ref": str(replay),
                        "visual_session_ref": str(visual),
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    index = ReportEvidenceReader().build_index(run_dir)

    refs_by_path = {ref.path: ref for ref in index.evidence_refs.values()}
    assert set(refs_by_path) == {
        "replays/sample-1/replay.json",
        "replays/sample-1/arena_visual_session/v1/manifest.json",
    }
    assert refs_by_path["replays/sample-1/replay.json"].artifact_role == "replay"
    assert refs_by_path["replays/sample-1/arena_visual_session/v1/manifest.json"].artifact_role == "visual_session"
    assert all(ref.ref_id and ref.ref_id.startswith("evidence://artifact/") for ref in refs_by_path.values())


@pytest.mark.io
def test_reader_ignores_legacy_replay_path_and_indexes_declared_replay_ref(tmp_path) -> None:
    run_dir = tmp_path / "run"
    sample_replay = run_dir / "replays" / "sample-level" / "replay.json"
    judge_replay = run_dir / "replays" / "judge-level" / "replay.json"
    sample_replay.parent.mkdir(parents=True)
    judge_replay.parent.mkdir(parents=True)
    sample_replay.write_text(json.dumps({"move_count": 2}), encoding="utf-8")
    judge_replay.write_text(json.dumps({"move_count": 3}), encoding="utf-8")
    (run_dir / "samples.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "task_id": "tictactoe",
                "replay_path": str(sample_replay),
                "judge_output": {
                    "artifacts": {"replay_ref": "replays/judge-level/replay.json"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    index = ReportEvidenceReader().build_index(run_dir)

    assert {ref.path for ref in index.evidence_refs.values()} == {"replays/judge-level/replay.json"}


@pytest.mark.io
def test_reader_indexes_remote_mmmu_image_url_without_query_leak(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    url = (
        "https://huggingface.co/datasets/modelscope/MMMU/resolve/main/images/"
        "validation_Agriculture_16_1.png?X-Amz-Signature=secret-token"
    )
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    (run_dir / "samples.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "validation_Agriculture_16",
                "task_id": "mmmu",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": url}},
                            {"type": "text", "text": "What is shown?"},
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    index = ReportEvidenceReader().build_index(run_dir)

    assert len(index.evidence_refs) == 1
    assert next(iter(index.evidence_refs)).startswith(f"evidence://media/{digest[:12]}-")
    ref = next(iter(index.evidence_refs.values()))
    assert ref.kind == "media"
    assert ref.path == f"external://sha256/{digest}"
    assert ref.sha256 == digest
    assert ref.mime_type == "image/png"
    assert ref.size_bytes is None
    assert ref.timestamp_iso is None
    assert ref.sample_id == "validation_Agriculture_16"
    assert ref.task_id == "mmmu"
    assert "X-Amz-Signature" not in json.dumps(ref.to_dict())
    assert "secret-token" not in json.dumps(ref.to_dict())
    assert ref.preview == {
        "source": "huggingface.co/datasets/modelscope/MMMU/resolve/main/images/validation_Agriculture_16_1.png"
    }


@pytest.mark.io
def test_reader_keeps_same_remote_media_url_for_multiple_samples(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    url = "https://example.test/assets/shared.png?token=secret"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    records = [
        {
            "sample_id": "sample-1",
            "task_id": "mmmu",
            "inputs": {"multi_modal_data": {"image": url}},
        },
        {
            "sample_id": "sample-2",
            "task_id": "mmmu",
            "inputs": {"multi_modal_data": {"image": url}},
        },
    ]
    (run_dir / "samples.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    index = ReportEvidenceReader().build_index(run_dir)

    refs = sorted(index.evidence_refs.values(), key=lambda ref: ref.sample_id or "")
    assert [ref.sample_id for ref in refs] == ["sample-1", "sample-2"]
    assert {ref.path for ref in refs} == {f"external://sha256/{digest}"}
    assert {ref.sha256 for ref in refs} == {digest}
    assert all(ref.ref_id and ref.ref_id.startswith("evidence://media/") for ref in refs)
    assert len({ref.ref_id for ref in refs}) == 2


@pytest.mark.io
def test_reader_indexes_nested_mmmu_multi_modal_image_dict_values(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    url = "https://example.test/assets/diagram.jpeg"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    (run_dir / "samples.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "sample": {
                    "task_id": "mmmu-nested",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": url}},
                            ],
                        }
                    ],
                    "inputs": {
                        "multi_modal_data": {
                            "image": [
                                {"url": url},
                                {"path": "https://example.test/assets/other.webp"},
                            ]
                        }
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    index = ReportEvidenceReader().build_index(run_dir)

    assert any(ref_id.startswith(f"evidence://media/{digest[:12]}-") for ref_id in index.evidence_refs)
    assert {ref.mime_type for ref in index.evidence_refs.values()} == {"image/jpeg", "image/webp"}
    assert all(ref.kind == "media" for ref in index.evidence_refs.values())
    assert all(ref.task_id == "mmmu-nested" for ref in index.evidence_refs.values())


@pytest.mark.io
def test_reader_indexes_top_level_multi_modal_scalar_image_url(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    url = "https://example.test/assets/top-level.png"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    (run_dir / "samples.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample-scalar",
                "task_id": "mmmu",
                "inputs": {"multi_modal_data": {"image": url}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    index = ReportEvidenceReader().build_index(run_dir)

    assert len(index.evidence_refs) == 1
    assert next(iter(index.evidence_refs)).startswith(f"evidence://media/{digest[:12]}-")
    ref = next(iter(index.evidence_refs.values()))
    assert ref.kind == "media"
    assert ref.path == f"external://sha256/{digest}"
    assert ref.mime_type == "image/png"


@pytest.mark.io
def test_reader_indexes_nested_multi_modal_scalar_image_dict(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    url = "https://example.test/assets/nested.webp"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    (run_dir / "samples.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample-dict",
                "sample": {
                    "task_id": "mmmu-nested",
                    "inputs": {"multi_modal_data": {"image": {"url": url}}},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    index = ReportEvidenceReader().build_index(run_dir)

    assert len(index.evidence_refs) == 1
    assert next(iter(index.evidence_refs)).startswith(f"evidence://media/{digest[:12]}-")
    ref = next(iter(index.evidence_refs.values()))
    assert ref.kind == "media"
    assert ref.path == f"external://sha256/{digest}"
    assert ref.mime_type == "image/webp"
    assert ref.task_id == "mmmu-nested"


@pytest.mark.io
def test_reader_does_not_serialize_data_image_media_evidence(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "samples.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "task_id": "mmmu",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AAAABBBB"},
                            }
                        ],
                    }
                ],
                "inputs": {"multi_modal_data": {"image": ["data:image/jpeg;base64,CCCCDDDD"]}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    index = ReportEvidenceReader().build_index(run_dir)

    serialized = json.dumps([ref.to_dict() for ref in index.evidence_refs.values()])
    assert "data:image" not in serialized
