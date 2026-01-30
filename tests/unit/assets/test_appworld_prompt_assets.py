from __future__ import annotations

import pytest

from gage_eval.assets.prompts.assets import PromptTemplateAsset
from gage_eval.assets.prompts.renderers import PromptContext
from gage_eval.registry import registry


@pytest.mark.fast
def test_appworld_prompt_asset_renders_api_docs_context() -> None:
    registry.auto_discover("prompts", "gage_eval.assets.prompts.catalog")
    asset = registry.get("prompts", "dut/appworld@v1")
    assert isinstance(asset, PromptTemplateAsset)

    renderer = asset.instantiate({"mode": "full"})
    sample = {
        "support_outputs": [
            {
                "api_docs_context": "# api_docs\n{}",
                "tool_documentation": "TOOLS",
            }
        ]
    }
    context = PromptContext(sample=sample, payload={"instruction": "Book a flight"})
    result = renderer.render(context)

    assert result.messages is not None
    content = result.messages[0]["content"]
    assert "I am your supervisor" in content
    assert "Real Task Instruction" in content
