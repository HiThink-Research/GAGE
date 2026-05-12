from __future__ import annotations

import asyncio

import pytest

from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.adapters.dut_agent import DUTAgentAdapter


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def aexecute(self, *, sample, payload, trace=None):
        self.calls.append({"sample": sample, "payload": payload, "trace": trace})
        return {"answer": "runtime"}


class FailingPromptRenderer:
    def __init__(self) -> None:
        self.render_calls = 0

    def render(self, context):
        self.render_calls += 1
        raise AssertionError("DUTAgentAdapter must not render prompts")


class FailingLegacyBackend:
    def __init__(self) -> None:
        self.invoke_calls = 0
        self.ainvoke_calls = 0
        self.shutdown_calls = 0

    def invoke(self, payload):
        self.invoke_calls += 1
        raise AssertionError("legacy backend invoke must not be called")

    async def ainvoke(self, payload):
        self.ainvoke_calls += 1
        raise AssertionError("legacy backend ainvoke must not be called")

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        raise AssertionError("legacy backend shutdown must not be called")


class ShutdownTrackingSandboxManager:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.mark.fast
def test_dut_agent_does_not_render_prompt_and_passes_payload_to_executor_unchanged() -> None:
    executor = RecordingExecutor()
    prompt_renderer = FailingPromptRenderer()
    adapter = DUTAgentAdapter(
        adapter_id="dut_agent",
        role_type="dut_agent",
        capabilities=(),
        prompt_renderer=prompt_renderer,
        executor_ref=executor,
    )
    trace = object()
    payload = {
        "sample": {
            "messages": [{"role": "user", "content": "hi"}],
            "prompt_context": {"tool_documentation": "DOCS"},
        },
        "trace": trace,
    }

    result = asyncio.run(adapter.ainvoke(payload, RoleAdapterState()))

    assert result == {"answer": "runtime"}
    assert prompt_renderer.render_calls == 0
    assert executor.calls == [
        {"sample": payload["sample"], "payload": payload, "trace": trace}
    ]


@pytest.mark.fast
def test_dut_agent_ignores_legacy_backend_when_executor_ref_is_present() -> None:
    executor = RecordingExecutor()
    backend = FailingLegacyBackend()
    adapter = DUTAgentAdapter(
        adapter_id="dut_agent",
        role_type="dut_agent",
        capabilities=(),
        agent_backend=backend,
        executor_ref=executor,
    )
    payload = {"sample": {"messages": [{"role": "user", "content": "hi"}]}}

    result = asyncio.run(adapter.ainvoke(payload, RoleAdapterState()))

    assert result == {"answer": "runtime"}
    assert backend.invoke_calls == 0
    assert backend.ainvoke_calls == 0


@pytest.mark.fast
def test_dut_agent_shutdown_only_releases_executor_resources() -> None:
    backend = FailingLegacyBackend()
    adapter_sandbox_manager = ShutdownTrackingSandboxManager()
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
        sandbox_manager=adapter_sandbox_manager,
        executor_ref=executor_ref,
    )

    adapter.shutdown()

    assert executor_sandbox_manager.shutdown_calls == 1
    assert backend.shutdown_calls == 0
    assert adapter_sandbox_manager.shutdown_calls == 0
