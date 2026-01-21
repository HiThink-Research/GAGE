from __future__ import annotations

import pytest

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
