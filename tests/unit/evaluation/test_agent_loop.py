import asyncio
from typing import Any

import pytest

from gage_eval.mcp import McpClient
from gage_eval.role.agent.loop import AgentLoop
from gage_eval.role.agent.tool_router import ToolRouter
from gage_eval.sandbox.base import ExecResult
from gage_eval.sandbox.manager import SandboxManager
from gage_eval.sandbox.provider import SandboxProvider, SandboxScope


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        if self.calls == 1:
            return {
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "run_shell", "arguments": "{\"command\": \"ls\"}"},
                    }
                ]
            }
        return {"answer": "done"}


class PlainThenToolBackend:
    def __init__(self):
        self.calls = 0
        self.payloads = []

    def invoke(self, payload):
        self.calls += 1
        self.payloads.append(dict(payload))
        if self.calls == 1:
            return {"answer": "I can help, please check your settings."}
        if self.calls == 2:
            return {
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "run_shell", "arguments": "{\"command\": \"ls\"}"},
                    }
                ]
            }
        return {"answer": "done"}


class PlainToolLoopBackend:
    def __init__(self):
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        if self.calls == 1:
            return {"answer": "I need a tool"}
        if self.calls == 2:
            return {
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "run_shell", "arguments": "{\"command\": \"ls\"}"},
                    }
                ]
            }
        return {"answer": "done"}


class RetryBudgetExceededBackend:
    def __init__(self):
        self.calls = 0
        self.tool_choice_required_payloads: list[Any] = []

    def invoke(self, payload):
        self.calls += 1
        self.tool_choice_required_payloads.append(payload.get("tool_choice"))
        return {"answer": "please call a tool", "raw_response": {"content": "please"}}


class BareCallRetryBackend:
    def __init__(self):
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        return {
            "answer": "call:respond{message:Hello Emma Kim! I can help with that.}",
            "raw_response": {"content": "call:respond"},
        }


class AsyncRetryBackend:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, payload):
        self.calls += 1
        return {"answer": "please call a tool", "raw_response": {"content": "please"}}


class AsyncBackend:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, payload):
        self.calls += 1
        if self.calls == 1:
            return {
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "run_shell", "arguments": "{\"command\": \"ls\"}"},
                    }
                ]
            }
        return {"answer": "done"}


class UsageBackend:
    def __init__(self):
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        if self.calls == 1:
            return {
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "run_shell", "arguments": "{\"command\": \"ls\"}"},
                    }
                ],
            }
        return {
            "answer": "done",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }


class FakeSandbox:
    def __init__(self, runtime_configs=None, resources=None):
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}

    def start(self, config):
        return {"sandbox_id": "fake"}

    def exec(self, command, timeout=30):
        return ExecResult(exit_code=0, stdout="ok", stderr="", duration_ms=1.0)

    def teardown(self):
        return None

    def is_alive(self, timeout_s=None):
        return True


class FailingFakeSandbox(FakeSandbox):
    def exec(self, command, timeout=30):
        return ExecResult(exit_code=127, stdout="", stderr="/bin/sh: 1: badcmd: not found\n", duration_ms=1.0)


@pytest.mark.fast
def test_agent_loop_tool_call_flow():
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    provider = SandboxProvider(
        manager,
        {"runtime": "fake"},
        SandboxScope(run_id="run", task_id="task", sample_id="sample"),
    )
    loop = AgentLoop(
        backend=FakeBackend(),
        tool_router=ToolRouter(),
        max_turns=3,
    )
    result = loop.run(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "run_shell", "parameters": {}}}],
        sandbox_config={"runtime": "fake"},
        sandbox_provider=provider,
    )
    provider.release()
    assert result["answer"] == "done"
    assert len(result["agent_trace"]) == 2
    assert result["agent_trace"][0]["trace_role"] == "tool"
    assert result["agent_trace"][1]["trace_role"] == "assistant"


@pytest.mark.fast
def test_agent_loop_accumulates_usage_across_turns():
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    provider = SandboxProvider(
        manager,
        {"runtime": "fake"},
        SandboxScope(run_id="run", task_id="task", sample_id="sample"),
    )
    loop = AgentLoop(
        backend=UsageBackend(),
        tool_router=ToolRouter(),
        max_turns=3,
    )

    result = loop.run(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "run_shell", "parameters": {}}}],
        sandbox_config={"runtime": "fake"},
        sandbox_provider=provider,
    )

    provider.release()
    assert result["usage"] == {
        "prompt_tokens": 8,
        "completion_tokens": 3,
        "total_tokens": 11,
    }


@pytest.mark.fast
def test_agent_loop_retries_when_required_tool_call_is_missing() -> None:
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    provider = SandboxProvider(
        manager,
        {"runtime": "fake"},
        SandboxScope(run_id="run", task_id="task", sample_id="sample"),
    )
    backend = PlainThenToolBackend()
    backend._force_tool_choice_mode = "first_turn"
    loop = AgentLoop(
        backend=backend,
        tool_router=ToolRouter(),
        max_turns=4,
    )

    result = loop.run(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "run_shell", "parameters": {}}}],
        tool_choice="auto",
        sandbox_config={"runtime": "fake"},
        sandbox_provider=provider,
    )

    provider.release()
    assert result["answer"] == "done"
    assert backend.calls == 3
    assert backend.payloads[0]["tool_choice"] == "required"
    assert backend.payloads[1]["tool_choice"] == "required"
    assert result["agent_trace"][0]["status"] == "retry_required_tool_call"
    assert result["agent_trace"][0]["output"]["error"] == "required_tool_call_missing"
    assert result["agent_trace"][1]["trace_role"] == "tool"
    assert result["agent_trace"][2]["trace_role"] == "assistant"


@pytest.mark.fast
def test_agent_loop_retries_until_budget_exhausted() -> None:
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    provider = SandboxProvider(
        manager,
        {"runtime": "fake"},
        SandboxScope(run_id="run", task_id="task", sample_id="sample"),
    )
    backend = RetryBudgetExceededBackend()
    backend._force_tool_choice_mode = "always"
    loop = AgentLoop(
        backend=backend,
        tool_router=ToolRouter(),
        max_turns=10,
        tool_call_retry_budget=3,
    )

    result = loop.run(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "run_shell", "parameters": {}}}],
        tool_choice="required",
        sandbox_config={"runtime": "fake"},
        sandbox_provider=provider,
    )
    provider.release()

    assert backend.calls == 3
    assert result["loop_exit_reason"] == "tool_call_retry_budget"
    assert result["answer"] == ""
    assert len(result["agent_trace"]) == 3
    assert result["agent_trace"][0]["status"] == "retry_required_tool_call"
    assert result["agent_trace"][1]["status"] == "retry_required_tool_call"
    assert result["agent_trace"][2]["status"] == "retry_required_tool_call"
    assert result["agent_trace"][0]["output"]["retry_count"] == 1
    assert result["agent_trace"][1]["output"]["retry_count"] == 2
    assert result["agent_trace"][2]["output"]["retry_count"] == 3
    events = result["observability_events"]
    missing_retry_events = [event for event in events if event.get("event") == "agent_retry_missing_tool_call"]
    assert len(missing_retry_events) == 3
    assert all(
        event["payload"]["answer_preview"] == "please call a tool"
        for event in missing_retry_events
    )
    assert all(
        event["payload"]["has_tool_call_tag"] is False and event["payload"]["has_function_tag"] is False
        for event in missing_retry_events
    )
    assert all(
        event["payload"]["backend_has_raw_response"] is True for event in missing_retry_events
    )
    assert any(event.get("event") == "agent_loop_exhausted" for event in events)
    exhausted = [event for event in events if event.get("event") == "agent_loop_exhausted"][0]
    assert exhausted["payload"]["reason"] == "tool_call_retry_budget"
    assert exhausted["payload"]["consecutive_retries"] == 3
    assert exhausted["payload"]["budget"] == 3
    assert not any(
        event.get("event") == "agent_loop_exhausted" and event["payload"].get("reason") == "max_turns"
        for event in events
    )
    assert all(tc == "required" for tc in backend.tool_choice_required_payloads[:3])


@pytest.mark.fast
def test_agent_loop_missing_tool_call_event_marks_bare_call_prefix() -> None:
    backend = BareCallRetryBackend()
    loop = AgentLoop(
        backend=backend,
        tool_router=ToolRouter(),
        max_turns=2,
        tool_call_retry_budget=1,
    )

    result = loop.run(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "respond", "parameters": {}}}],
        tool_choice="required",
    )

    retry_events = [
        event for event in result["observability_events"] if event.get("event") == "agent_retry_missing_tool_call"
    ]
    assert backend.calls == 1
    assert retry_events[0]["payload"]["has_bare_call_prefix"] is True
    assert retry_events[0]["payload"]["has_minimax_tag"] is False


@pytest.mark.fast
def test_agent_loop_retries_clear_after_tool_call() -> None:
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    provider = SandboxProvider(
        manager,
        {"runtime": "fake"},
        SandboxScope(run_id="run", task_id="task", sample_id="sample"),
    )
    backend = PlainToolLoopBackend()
    backend._force_tool_choice_mode = "first_turn"
    loop = AgentLoop(
        backend=backend,
        tool_router=ToolRouter(),
        max_turns=5,
        tool_call_retry_budget=2,
    )

    result = loop.run(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "run_shell", "parameters": {}}}],
        tool_choice="auto",
        sandbox_config={"runtime": "fake"},
        sandbox_provider=provider,
    )
    provider.release()

    assert backend.calls == 3
    assert result["loop_exit_reason"] is None
    assert len(result["agent_trace"]) == 3


@pytest.mark.fast
def test_agent_loop_retries_by_max_turns_when_budget_not_hit() -> None:
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    provider = SandboxProvider(
        manager,
        {"runtime": "fake"},
        SandboxScope(run_id="run", task_id="task", sample_id="sample"),
    )
    backend = RetryBudgetExceededBackend()
    backend._force_tool_choice_mode = "always"
    loop = AgentLoop(
        backend=backend,
        tool_router=ToolRouter(),
        max_turns=2,
        tool_call_retry_budget=10,
    )

    result = loop.run(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "run_shell", "parameters": {}}}],
        tool_choice="required",
        sandbox_config={"runtime": "fake"},
        sandbox_provider=provider,
    )
    provider.release()

    assert backend.calls == 2
    assert result["loop_exit_reason"] == "max_turns"
    assert len(result["observability_events"]) == 3
    retry_missing_events = [
        event
        for event in result["observability_events"]
        if event.get("event") == "agent_retry_missing_tool_call"
    ]
    exhausted = [event for event in result["observability_events"] if event["event"] == "agent_loop_exhausted"][
        0
    ]
    assert len(retry_missing_events) == 2
    assert exhausted["event"] == "agent_loop_exhausted"
    assert exhausted["payload"]["reason"] == "max_turns"


@pytest.mark.fast
def test_agent_loop_async_tool_call_flow() -> None:
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    provider = SandboxProvider(
        manager,
        {"runtime": "fake"},
        SandboxScope(run_id="run", task_id="task", sample_id="sample"),
    )
    loop = AgentLoop(
        backend=AsyncBackend(),
        tool_router=ToolRouter(),
        max_turns=3,
    )
    result = asyncio.run(
        loop.arun(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "run_shell", "parameters": {}}}],
            sandbox_config={"runtime": "fake"},
            sandbox_provider=provider,
        )
    )
    provider.release()
    assert result["answer"] == "done"
    assert len(result["agent_trace"]) == 2
    assert result["agent_trace"][0]["trace_role"] == "tool"
    assert result["agent_trace"][1]["trace_role"] == "assistant"


@pytest.mark.fast
def test_agent_loop_async_retries_until_budget_exhausted() -> None:
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    provider = SandboxProvider(
        manager,
        {"runtime": "fake"},
        SandboxScope(run_id="run", task_id="task", sample_id="sample"),
    )
    backend = AsyncRetryBackend()
    loop = AgentLoop(
        backend=backend,
        tool_router=ToolRouter(),
        max_turns=10,
        tool_call_retry_budget=2,
    )
    result = asyncio.run(
        loop.arun(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "run_shell", "parameters": {}}}],
            sandbox_config={"runtime": "fake"},
            sandbox_provider=provider,
            tool_choice="required",
        )
    )
    provider.release()

    assert backend.calls == 2
    assert result["loop_exit_reason"] == "tool_call_retry_budget"
    assert result["answer"] == ""
    assert len(result["agent_trace"]) == 2
    assert result["agent_trace"][0]["output"]["retry_count"] == 1
    assert result["agent_trace"][1]["output"]["retry_count"] == 2
    events = result["observability_events"]
    assert any(event["event"] == "agent_loop_exhausted" for event in events)
    exhausted = [event for event in events if event["event"] == "agent_loop_exhausted"][0]
    assert exhausted["payload"]["reason"] == "tool_call_retry_budget"


@pytest.mark.fast
def test_agent_loop_records_failed_tool_trace_for_nonzero_exit_code():
    class FailingBackend:
        def __init__(self):
            self.calls = 0

        def invoke(self, payload):
            self.calls += 1
            if self.calls == 1:
                return {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "run_shell", "arguments": "{\"command\": \"badcmd\"}"},
                        }
                    ]
                }
            return {"answer": "done"}

    manager = SandboxManager()
    manager.register_runtime("fake", FailingFakeSandbox)
    provider = SandboxProvider(
        manager,
        {"runtime": "fake"},
        SandboxScope(run_id="run", task_id="task", sample_id="sample"),
    )
    loop = AgentLoop(
        backend=FailingBackend(),
        tool_router=ToolRouter(),
        max_turns=3,
    )
    result = loop.run(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "run_shell", "parameters": {}}}],
        sandbox_config={"runtime": "fake"},
        sandbox_provider=provider,
    )
    provider.release()
    assert result["agent_trace"][0]["trace_role"] == "tool"
    assert result["agent_trace"][0]["status"] == "failed"
    assert result["agent_trace"][0]["output"]["exit_code"] == 127


@pytest.mark.fast
def test_agent_loop_records_resolved_tool_for_meta_calls() -> None:
    class MetaBackend:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, payload):
            self.calls += 1
            if self.calls == 1:
                return {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "call_spotify",
                                "arguments": {"endpoint": "search", "params": {"q": "hi"}},
                            },
                        }
                    ]
                }
            return {"answer": "done"}

    executed = {}

    def executor(name, arguments):
        executed["name"] = name
        executed["arguments"] = arguments
        return {"status": "ok"}

    client = McpClient(
        mcp_client_id="mcp_main",
        transport="stub",
        endpoint="http://example.com",
        params={"executor": executor, "tools": []},
    )
    router = ToolRouter(mcp_clients={"mcp_main": client})
    tools = [
        {
            "type": "function",
            "function": {
                "name": "call_spotify",
                "description": "Call spotify endpoints.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "endpoint": {"type": "string"},
                        "params": {"type": "object"},
                    },
                    "required": ["endpoint"],
                },
            },
            "x-gage": {
                "meta_tool": True,
                "app_name": "spotify",
                "allowed_endpoints": ["search"],
                "mcp_client_id": "mcp_main",
            },
        }
    ]

    loop = AgentLoop(
        backend=MetaBackend(),
        tool_router=router,
        max_turns=3,
    )
    result = loop.run(messages=[{"role": "user", "content": "hi"}], tools=tools)
    assert executed["name"] == "spotify__search"
    assert executed["arguments"] == {"q": "hi"}
    tool_step = result["agent_trace"][0]
    assert tool_step["trace_role"] == "tool"
    assert tool_step["resolved_tool"] == "spotify__search"
