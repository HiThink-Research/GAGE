from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from gage_eval.evaluation.cache import EvalCache


@pytest.mark.io
def test_eval_cache_root_journal_uses_json_default(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=str(tmp_path), run_id="json-default")

    cache.write_sample("sample-1", {"path": tmp_path, "value": 1})

    records = list(cache.iter_samples())
    assert len(records) == 1
    assert records[0]["sample_id"] == "sample-1"
    assert records[0]["path"] == str(tmp_path)


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
