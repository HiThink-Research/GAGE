from __future__ import annotations

import asyncio
import threading

import pytest

from gage_eval.assets.prompts.renderers import PromptContext, PromptRenderResult, PromptRenderer
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.adapters.dut_agent import DUTAgentAdapter
from gage_eval.role.agent.backends.model_backend import ModelBackend


class FakeBackend:
    def invoke(self, payload):
        return {"answer": "ok"}


class AsyncModelOnlyBackend:
    def __init__(self) -> None:
        self.thread_id: int | None = None

    async def ainvoke(self, payload):
        self.thread_id = threading.get_ident()
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
            "prompt_context": {"tool_documentation": "DOCS"},
        }
    }

    result = asyncio.run(adapter.ainvoke(payload, RoleAdapterState()))

    assert result["system_prompt"].startswith("System:")
    assert "DOCS" in result["system_prompt"]


@pytest.mark.fast
def test_dut_agent_awaits_async_model_backend_in_current_thread() -> None:
    wrapped_backend = AsyncModelOnlyBackend()
    adapter = DUTAgentAdapter(
        adapter_id="dut_agent",
        role_type="dut_agent",
        capabilities=(),
        agent_backend=ModelBackend({"backend": wrapped_backend}),
    )
    payload = {"sample": {"messages": [{"role": "user", "content": "hi"}]}}
    observed_thread_ids: list[int] = []

    async def _run() -> None:
        observed_thread_ids.append(threading.get_ident())
        result = await adapter.ainvoke(payload, RoleAdapterState())

        assert result["answer"] == "ok"

    asyncio.run(_run())

    assert wrapped_backend.thread_id == observed_thread_ids[0]
