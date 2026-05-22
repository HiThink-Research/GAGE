from __future__ import annotations

import json

import pytest

from gage_eval.evaluation.cache import EvalCache
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.report import ReportStep


@pytest.mark.io
def test_report_pack_end_to_end_redacts_secrets(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_REPORT_PACK", "1")
    cache = EvalCache(base_dir=tmp_path, run_id="secret-pack")
    cache.write_sample("sample", {"sample": {"prompt": "Authorization: Bearer abc123"}, "status": "failed"})
    report = ReportStep(auto_eval_step=None, cache_store=cache)
    report.record_execution_summary({"headers": {"Authorization": "Bearer abc123"}})

    report.finalize(ObservabilityTrace(run_id="secret-pack"))

    pack = tmp_path / "secret-pack" / "report_pack"
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in pack.rglob("*") if path.is_file())
    assert "Bearer abc123" not in serialized
    assert "<redacted:" in serialized
    assert json.loads((pack / "diagnostics.json").read_text(encoding="utf-8"))["report_pack_status"]


@pytest.mark.io
def test_legacy_summary_redacts_execution_summary(tmp_path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="secret-summary")
    report = ReportStep(auto_eval_step=None, cache_store=cache)
    report.record_execution_summary({"headers": {"Authorization": "Bearer abc123"}})

    report.finalize(ObservabilityTrace(run_id="secret-summary"))

    serialized = (tmp_path / "secret-summary" / "summary.json").read_text(encoding="utf-8")
    assert "Bearer abc123" not in serialized
    assert "<redacted:" in serialized
