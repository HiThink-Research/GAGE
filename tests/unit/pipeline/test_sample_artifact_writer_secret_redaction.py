from __future__ import annotations

import json

import pytest

from gage_eval.evaluation.cache import EvalCache
from gage_eval.pipeline.sample_artifact_writer import SampleArtifactWriter


@pytest.mark.fast
def test_sample_artifact_writer_redacts_report_visible_payloads(tmp_path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="redacted-sample")
    writer = SampleArtifactWriter(cache_store=cache)

    result = writer.write_sample_record(
        {
            "task_id": "task",
            "sample_id": "sample",
            "sample": {"id": "sample", "prompt": "Authorization: Bearer abc123"},
            "trial_results": [],
            "aggregate": {"error": "password=secret"},
            "effective_config": {"api_key": "sk-abcdefghijklmnopqrstuvwxyz1234567890"},
        }
    )

    sample_record = json.loads((tmp_path / "redacted-sample" / result.sample_record_ref.path).read_text(encoding="utf-8"))
    cache_payload = json.loads((tmp_path / "redacted-sample" / "samples.jsonl").read_text(encoding="utf-8"))
    serialized = json.dumps([sample_record, cache_payload], ensure_ascii=False)
    assert "Bearer abc123" not in serialized
    assert "password=secret" not in serialized
    assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in serialized
    assert "<redacted:" in serialized
