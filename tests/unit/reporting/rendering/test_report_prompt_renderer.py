from __future__ import annotations

import pytest

from gage_eval.reporting.contracts import ReportContext
from gage_eval.reporting.rendering.prompt_renderer import PromptRenderer


@pytest.mark.fast
def test_prompt_renderer_instructs_evidence_bounded_reading() -> None:
    prompt = PromptRenderer().render(ReportContext.minimal("run").to_dict())

    assert "report_context.json" in prompt
    assert "EvidenceRef" in prompt
    assert "Do not invent" in prompt
    assert "do not read unredacted raw trace" in prompt
