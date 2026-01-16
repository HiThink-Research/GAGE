from __future__ import annotations

import asyncio

import pytest

from gage_eval.assets.prompts.renderers import PromptContext, PromptRenderResult, PromptRenderer
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.adapters.dut_agent import DUTAgentAdapter


class FakeBackend:
    def invoke(self, payload):
        return {"answer": "ok"}


class ToolDocPrompt(PromptRenderer):
    def render(self, context: PromptContext) -> PromptRenderResult:
        tool_doc = context.to_mapping().get("tool_documentation") or ""
        return PromptRenderResult(prompt=f"System: {tool_doc}")


@pytest.mark.fast
def test_dut_agent_persists_system_prompt() -> None:
    adapter = DUTAgentAdapter(
        adapter_id="dut_agent",
        role_type="dut_agent",
        capabilities=(),
        agent_backend=FakeBackend(),
        prompt_renderer=ToolDocPrompt(),
    )
    payload = {
        "sample": {
            "messages": [{"role": "user", "content": "hi"}],
            "support_outputs": [{"tool_documentation": "DOCS"}],
        }
    }

    result = asyncio.run(adapter.ainvoke(payload, RoleAdapterState()))

    assert result["system_prompt"].startswith("System:")
    assert "DOCS" in result["system_prompt"]
