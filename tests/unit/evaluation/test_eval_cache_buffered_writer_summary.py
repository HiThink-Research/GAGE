from __future__ import annotations

import json

import pytest

from gage_eval.evaluation.cache import EvalCache


@pytest.mark.fast
def test_cache_close_patches_final_buffered_writer_summary(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GAGE_EVAL_ENABLE_BUFFERED_WRITER", "1")
    monkeypatch.setenv("GAGE_EVAL_BUFFER_DURABILITY_POLICY", "interval")
    monkeypatch.setenv("GAGE_EVAL_BUFFER_FSYNC_EVERY_FLUSHES", "99")
    monkeypatch.setenv("GAGE_EVAL_BUFFER_FSYNC_EVERY_S", "999")
    cache = EvalCache(base_dir=tmp_path, run_id="buffered-summary")

    cache.write_sample("s1", {"value": 1}, namespace="default")
    summary_path = cache.write_summary({"ok": True})
    initial = json.loads(summary_path.read_text(encoding="utf-8"))

    assert initial["buffered_writer_flush_count"] == 1
    assert initial["buffered_writer_fsync_count"] == 0
    assert initial["buffered_writer_durability_policy"] == "interval"

    cache.close()

    patched = json.loads(summary_path.read_text(encoding="utf-8"))

    assert patched["buffered_writer_flush_count"] == 1
    assert patched["buffered_writer_fsync_count"] == 1
    assert patched["buffered_writer_active_namespaces"] == 1
