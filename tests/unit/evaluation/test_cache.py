from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from gage_eval.evaluation.cache import EvalCache


@pytest.mark.io
def test_eval_cache_root_journal_uses_json_default(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=str(tmp_path), run_id="json-default")

    artifact = cache.write_sample("sample-1", {"path": tmp_path, "value": 1})

    records = list(cache.iter_samples())
    assert len(records) == 1
    assert records[0]["sample_id"] == "sample-1"
    assert records[0]["path"] == str(tmp_path)
    assert artifact.name == "sample-1.json"


@pytest.mark.io
def test_eval_cache_preserves_short_sample_id_filename(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=str(tmp_path), run_id="short-id")

    artifact = cache.write_sample("readable-sample_1", {"value": 1}, namespace="ns")

    assert artifact.name == "readable-sample_1.json"
    assert artifact.exists()


@pytest.mark.io
def test_eval_cache_keeps_245_byte_sample_id_untruncated(tmp_path: Path) -> None:
    sample_id = "a" * 245

    sanitized = EvalCache._sanitize_sample_id(sample_id)

    assert sanitized == sample_id
    assert len(f"{sanitized}.json".encode("utf-8")) <= 255


@pytest.mark.io
def test_eval_cache_truncates_over_245_byte_sample_id_with_hash(tmp_path: Path) -> None:
    sample_id = "b" * 260

    sanitized = EvalCache._sanitize_sample_id(sample_id)

    assert sanitized.startswith("b" * 237)
    assert len(sanitized.rsplit("_", 1)[-1]) == 8
    assert len(f"{sanitized}.json".encode("utf-8")) <= 255


@pytest.mark.io
def test_eval_cache_redacts_samples_jsonl_and_sample_artifact(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=str(tmp_path), run_id="redacted-sample")

    cache.write_sample(
        "sample-1",
        {
            "model_output": {
                "tool_call": {
                    "arguments": '{"username":"user@example.com","password":"password123"}'
                }
            }
        },
        namespace="ns",
    )

    samples_text = cache.samples_jsonl.read_text(encoding="utf-8")
    artifact_text = next((cache.samples_dir / "ns").glob("*.json")).read_text(encoding="utf-8")
    combined = samples_text + artifact_text
    assert "user@example.com" not in combined
    assert "password123" not in combined
    assert "<redacted:email>" in combined
    assert "<redacted:secret>" in combined


@pytest.mark.io
def test_eval_cache_root_journal_handles_concurrent_writes(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=str(tmp_path), run_id="concurrent")
    total_writes = 32
    barrier = threading.Barrier(total_writes)
    errors: list[Exception] = []

    def _worker(index: int) -> None:
        try:
            barrier.wait()
            cache.write_sample(
                f"sample-{index}",
                {"value": index},
                namespace="ns",
            )
        except Exception as exc:  # pragma: no cover - defensive in threads
            errors.append(exc)

    threads = [
        threading.Thread(target=_worker, args=(index,))
        for index in range(total_writes)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert cache.sample_count == total_writes

    lines = cache.samples_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == total_writes

    records = [json.loads(line) for line in lines]
    assert sorted(record["sample_id"] for record in records) == sorted(
        f"sample-{index}" for index in range(total_writes)
    )

    artifact_files = sorted((cache.samples_dir / "ns").glob("*.json"))
    assert len(artifact_files) == total_writes
