from __future__ import annotations

import pytest

from gage_eval.reporting.contracts import ReportContext
from gage_eval.reporting.rendering.markdown_renderer import MarkdownRenderer


@pytest.mark.fast
def test_markdown_renderer_summarizes_context_without_secret() -> None:
    markdown = MarkdownRenderer().render(ReportContext.minimal("run").to_dict())

    assert "# GAGE Run Report" in markdown
    assert "run" in markdown
    assert "Bearer " not in markdown
