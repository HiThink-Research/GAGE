from __future__ import annotations

import pytest

from gage_eval.evaluation.support_artifacts import record_support_output
from gage_eval.assets.prompts.renderers import JinjaChatPromptRenderer, PromptContext


@pytest.mark.fast
def test_prompt_renderer_injects_tool_documentation() -> None:
    renderer = JinjaChatPromptRenderer(template="{{ tool_documentation }}")
    context = PromptContext(
        sample={"support_outputs": [{"tool_documentation": "DOCS", "tool_documentation_meta": {"apps": 1}}]},
        payload={},
    )
    result = renderer.render(context)

    assert result.messages is not None
    assert result.messages[0]["content"] == "DOCS"


@pytest.mark.fast
def test_prompt_renderer_reads_tool_documentation_from_support_artifacts() -> None:
    renderer = JinjaChatPromptRenderer(template="{{ tool_documentation }}")
    sample = {}
    record_support_output(
        sample,
        slot_id="support:00:toolchain_main",
        adapter_id="toolchain_main",
        output={"tool_documentation": "ARTIFACT_DOCS", "tool_documentation_meta": {"apps": 2}},
    )
    sample.pop("support_outputs", None)
    context = PromptContext(sample=sample, payload={})

    result = renderer.render(context)

    assert result.messages is not None
    assert result.messages[0]["content"] == "ARTIFACT_DOCS"
