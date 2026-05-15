from __future__ import annotations

import pytest

from gage_eval.reporting.contracts import ReportContext
from gage_eval.reporting.rendering.static_renderer import StaticReportRenderer


@pytest.mark.io
def test_static_report_html_has_local_interaction_hooks() -> None:
    html = StaticReportRenderer().render(ReportContext.minimal("run").to_dict())

    assert "<details" in html
    assert "data-filter-target" in html
    assert "report.js" in html
