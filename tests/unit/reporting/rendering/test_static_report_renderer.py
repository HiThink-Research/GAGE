from __future__ import annotations

import pytest

from gage_eval.reporting.contracts import ReportContext
from gage_eval.reporting.rendering.static_renderer import StaticReportRenderer


@pytest.mark.fast
def test_static_renderer_outputs_required_sections_without_cdn() -> None:
    context = ReportContext.minimal("run").to_dict()
    html = StaticReportRenderer().render(context)

    for text in [
        "Run Overview",
        "Metrics",
        "Attention Cases",
        "Case Details",
        "Outliers",
        "Failure Analysis",
        "Evidence",
        "Scenario Profiles",
        "Methodology",
    ]:
        assert text in html
    assert 'src="https://' not in html
    assert "gage.report_context" in html


@pytest.mark.fast
def test_static_renderer_escapes_html() -> None:
    context = ReportContext.minimal("run").to_dict()
    context["headline"]["one_line_summary"] = "<script>alert(1)</script>"

    html = StaticReportRenderer().render(context)

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
