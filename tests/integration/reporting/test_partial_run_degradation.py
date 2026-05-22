from __future__ import annotations

import pytest

from gage_eval.reporting.contracts import ReportContext
from gage_eval.reporting.persistence.pack_builder import ReportPackBuilder


@pytest.mark.io
def test_report_pack_degraded_context_still_writes_diagnostics(tmp_path) -> None:
    context = ReportContext.minimal("partial")
    payload = context.to_dict()
    payload["diagnostics"]["report_pack_status"] = "degraded"
    context = ReportContext.from_dict(payload)

    result = ReportPackBuilder().write(tmp_path, context)

    assert result["report_pack_status"] == "degraded"
    assert (tmp_path / "report_pack" / "diagnostics.json").exists()
