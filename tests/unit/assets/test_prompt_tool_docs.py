from __future__ import annotations

import pytest

from gage_eval.assets.prompts.renderers import JinjaChatPromptRenderer, PromptContext


@pytest.mark.fast
def test_prompt_renderer_injects_tool_documentation() -> None:
    renderer = JinjaChatPromptRenderer(template="{{ tool_documentation }}")
    context = PromptContext(
        sample={"prompt_context": {"tool_documentation": "DOCS", "tool_documentation_meta": {"apps": 1}}},
        payload={},
    )
    result = renderer.render(context)

    assert result.messages is not None
    assert result.messages[0]["content"] == "DOCS"


@pytest.mark.fast
def test_prompt_renderer_reads_tool_documentation_from_payload_prompt_context() -> None:
    renderer = JinjaChatPromptRenderer(template="{{ tool_documentation }}")
    context = PromptContext(
        sample={},
        payload={"prompt_context": {"tool_documentation": "PAYLOAD_DOCS", "tool_documentation_meta": {"apps": 2}}},
    )

    result = renderer.render(context)

    assert result.messages is not None
    assert result.messages[0]["content"] == "PAYLOAD_DOCS"


@pytest.mark.fast
def test_prompt_renderer_ignores_legacy_support_outputs_for_tool_docs() -> None:
    renderer = JinjaChatPromptRenderer(template="{{ tool_documentation }}")
    context = PromptContext(
        sample={"support_outputs": [{"tool_documentation": "LEGACY_DOCS"}]},
        payload={},
    )

    result = renderer.render(context)

    assert result.messages == [] or result.messages is None
