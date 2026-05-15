from __future__ import annotations

import json

import pytest

from gage_eval.evaluation.cache import EvalCache
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.report import ReportStep


@pytest.mark.io
def test_report_step_finalize_writes_summary_and_report_pack(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_REPORT_PACK", "1")
    cache = EvalCache(base_dir=tmp_path, run_id="finalize-e2e")
    cache.write_sample("sample", {"sample": {"id": "sample"}, "status": "completed"})

    payload = ReportStep(auto_eval_step=None, cache_store=cache).finalize(ObservabilityTrace(run_id="finalize-e2e"))

    assert payload["sample_count"] == 1
    assert (tmp_path / "finalize-e2e" / "summary.json").exists()
    assert (tmp_path / "finalize-e2e" / "report_pack" / "report.html").exists()
    summary = json.loads((tmp_path / "finalize-e2e" / "summary.json").read_text(encoding="utf-8"))
    assert summary["report_pack"]["status"] == "completed"
    diagnostics = json.loads((tmp_path / "finalize-e2e" / "report_pack" / "diagnostics.json").read_text(encoding="utf-8"))
    warning_codes = {item["code"] for item in diagnostics.get("warnings", [])}
    assert "report_pack.summary_missing" not in warning_codes
