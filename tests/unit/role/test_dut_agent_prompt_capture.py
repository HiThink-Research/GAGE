from __future__ import annotations

import asyncio
import threading

import pytest

import gage_eval.role.adapters.dut_agent as dut_agent_module
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


class ShutdownTrackingBackend:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def invoke(self, payload):
        return {"answer": "ok"}

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class ShutdownTrackingSandboxManager:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


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


@pytest.mark.fast
def test_dut_agent_shutdown_releases_backend_and_sandbox_resources() -> None:
    backend = ShutdownTrackingBackend()
    sandbox_manager = ShutdownTrackingSandboxManager()
    executor_sandbox_manager = ShutdownTrackingSandboxManager()
    executor_ref = type(
        "_ExecutorRef",
        (),
        {
            "resource_manager": type(
                "_ResourceManager",
                (),
                {"_sandbox_manager": executor_sandbox_manager},
            )()
        },
    )()
    adapter = DUTAgentAdapter(
        adapter_id="dut_agent",
        role_type="dut_agent",
        capabilities=(),
        agent_backend=backend,
        sandbox_manager=sandbox_manager,
        executor_ref=executor_ref,
    )

    adapter.shutdown()

    assert backend.shutdown_calls == 1
    assert sandbox_manager.shutdown_calls == 1
    assert executor_sandbox_manager.shutdown_calls == 1


@pytest.mark.fast
def test_dut_agent_passes_max_total_invalid_tool_calls_to_agent_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _CapturingAgentLoop:
        def __init__(
            self,
            backend,
            tool_router,
            *,
            max_turns: int = 8,
            tool_call_retry_budget: int = 3,
            max_total_invalid_tool_calls: int = 20,
            pre_hooks=None,
            post_hooks=None,
        ) -> None:
            captured["backend"] = backend
            captured["max_turns"] = max_turns
            captured["tool_call_retry_budget"] = tool_call_retry_budget
            captured["max_total_invalid_tool_calls"] = max_total_invalid_tool_calls

        async def arun(self, **kwargs):
            return {"answer": "ok", "agent_trace": []}

    monkeypatch.setattr(dut_agent_module, "AgentLoop", _CapturingAgentLoop)

    adapter = DUTAgentAdapter(
        adapter_id="dut_agent",
        role_type="dut_agent",
        capabilities=(),
        agent_backend=FakeBackend(),
        max_turns=9,
        tool_call_retry_budget=4,
        max_total_invalid_tool_calls=12,
    )

    result = asyncio.run(
        adapter.ainvoke(
            {"sample": {"messages": [{"role": "user", "content": "hi"}]}},
            RoleAdapterState(),
        )
    )

    assert result["answer"] == "ok"
    assert captured["max_turns"] == 9
    assert captured["tool_call_retry_budget"] == 4
    assert captured["max_total_invalid_tool_calls"] == 12


@pytest.mark.fast
def test_dut_agent_rejects_non_positive_max_total_invalid_tool_calls() -> None:
    with pytest.raises(ValueError, match="max_total_invalid_tool_calls"):
        DUTAgentAdapter(
            adapter_id="dut_agent",
            role_type="dut_agent",
            capabilities=(),
            agent_backend=FakeBackend(),
            max_total_invalid_tool_calls=0,
        )
