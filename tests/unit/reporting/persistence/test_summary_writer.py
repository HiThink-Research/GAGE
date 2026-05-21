from __future__ import annotations

import json

import pytest

from gage_eval.evaluation.cache import EvalCache
from gage_eval.reporting.persistence.summary_writer import SummaryWriter


@pytest.mark.fast
def test_summary_writer_preserves_v1_fields_and_adds_report_pack(tmp_path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="summary-writer")

    path = SummaryWriter().write(
        cache,
        {
            "run": {"run_id": "summary-writer"},
            "metrics": [],
            "sample_count": 0,
            "runtime_health": {"sample_count": 0},
        },
        report_pack_diagnostics={"report_pack_status": "completed"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["sample_count"] == 0
    assert payload["runtime_health"]["sample_count"] == 0
    assert payload["report_pack"]["status"] == "completed"
