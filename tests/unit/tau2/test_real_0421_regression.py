from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gage_eval.role.agent.backends.model_backend import ModelBackend
from gage_eval.role.agent.loop import AgentLoop
from gage_eval.role.agent.tool_router import ToolRouter
from gage_eval.sandbox.tau2_runtime import CANONICAL, parse_terminal_signal


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "tau2" / "runtime_samples"

RESPOND_TOOL = {
    "type": "function",
    "function": {
        "name": "respond",
        "description": "Send a message to the tau2 user simulator.",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
    "x-gage": {"final_answer_key": "final_answer"},
}


class _StaticTextModel:
    def __init__(self, text: str, *, config: dict[str, Any] | None = None) -> None:
        self.text = text
        self.calls = 0
        self.payloads: list[dict[str, Any]] = []
        self.config = config or {}

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        self.payloads.append(dict(payload))
        return {
            "answer": self.text,
            "raw_response": {"outputs": [{"text": self.text}]},
        }


class _RespondSandbox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def is_alive(self) -> bool:
        return True

    def exec_tool(self, name: str, arguments: Any) -> dict[str, Any]:
        self.calls.append((name, arguments))
        message = arguments["message"] if isinstance(arguments, dict) else str(arguments)
        return {"final_answer": message, "user_message": message}


class _Provider:
    def __init__(self) -> None:
        self.sandbox = _RespondSandbox()

    def get_handle(self) -> SimpleNamespace:
        return SimpleNamespace(sandbox=self.sandbox, runtime_handle={})


def _fixture_text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def _fixture_json(name: str) -> Any:
    return json.loads(_fixture_text(name))


@pytest.mark.fast
def test_real_0421_harmony_xml_response_parses_as_respond_tool_call() -> None:
    text = _fixture_text("0421_tau_harmony_xml_respond_response.txt")
    model = _StaticTextModel(text)
    backend = ModelBackend({"backend": model})

    result = backend.invoke(
        {
            "messages": [{"role": "user", "content": "My mobile data is not working."}],
            "tools": [RESPOND_TOOL],
            "tool_choice": "required",
        }
    )

    assert model.calls == 1
    assert model.payloads[0]["tool_choice"] == "required"
    assert result["answer"] == ""
    assert result["raw_answer"] == text
    assert len(result["tool_calls"]) == 1
    call = result["tool_calls"][0]
    assert call["function"]["name"] == "respond"
    assert "phone number" in call["function"]["arguments"]["message"].lower()


@pytest.mark.fast
def test_real_0421_tau2_think_tail_bare_json_response_parses_as_respond_tool_call() -> None:
    text = _fixture_text("0421_tau2_think_tail_bare_json_respond_response.txt")
    model = _StaticTextModel(text)
    backend = ModelBackend({"backend": model})

    result = backend.invoke(
        {
            "messages": [{"role": "user", "content": "My mobile data is not working."}],
            "tools": [RESPOND_TOOL],
            "tool_choice": "required",
        }
    )

    assert result["answer"] == ""
    assert result["raw_answer"] == text
    assert len(result["tool_calls"]) == 1
    call = result["tool_calls"][0]
    assert call["function"]["name"] == "respond"
    assert "verify your account" in call["function"]["arguments"]["message"].lower()


@pytest.mark.fast
def test_real_0421_gemma4_airline_bare_call_response_parses_as_respond_tool_call() -> None:
    text = _fixture_text("0421_gemma4_airline_bare_call_respond_response.txt")
    model = _StaticTextModel(text)
    backend = ModelBackend({"backend": model, "tool_format": "gemma4"})

    result = backend.invoke(
        {
            "messages": [{"role": "user", "content": "I want to cancel reservation EHGLP3."}],
            "tools": [RESPOND_TOOL],
            "tool_choice": "required",
        }
    )

    assert result["answer"] == ""
    assert result["raw_answer"] == text
    assert len(result["tool_calls"]) == 1
    call = result["tool_calls"][0]
    assert call["function"]["name"] == "respond"
    assert "reservation ehglp3" in call["function"]["arguments"]["message"].lower()


@pytest.mark.fast
def test_real_0421_qwen_gpt_plain_text_backend_wraps_and_strips_think_tail() -> None:
    text = _fixture_text("0421_qwen_gpt_airline_plain_text_response.txt")
    model = _StaticTextModel(text, config={"model_path": "/mnt/models/qwen3_6_35B"})
    backend = ModelBackend(
        {
            "backend": model,
            "plain_text_response_tool": "respond",
            "plain_text_response_formats": ["qwen"],
        }
    )

    result = backend.invoke(
        {
            "messages": [{"role": "user", "content": "I want to cancel reservation AAAAAA."}],
            "tools": [RESPOND_TOOL],
            "tool_choice": "required",
        }
    )

    assert result["plain_text_response_wrapped"] is True
    assert result["answer"] == ""
    assert result["raw_answer"] == text
    call = result["tool_calls"][0]
    assert call["function"]["name"] == "respond"
    wrapped_message = call["function"]["arguments"]["message"]
    assert "</think>" not in wrapped_message
    assert "reason for cancellation" in wrapped_message.lower()


@pytest.mark.fast
def test_real_0421_qwen_gpt_plain_text_response_reaches_respond_without_retry() -> None:
    text = _fixture_text("0421_qwen_gpt_airline_plain_text_response.txt")
    model = _StaticTextModel(text, config={"model_path": "/mnt/models/qwen3_6_35B"})
    provider = _Provider()
    loop = AgentLoop(
        backend=ModelBackend(
            {
                "backend": model,
                "plain_text_response_tool": "respond",
                "plain_text_response_formats": ["qwen"],
            }
        ),
        tool_router=ToolRouter(),
        max_turns=200,
        tool_call_retry_budget=3,
    )

    result = loop.run(
        messages=[{"role": "user", "content": "I want to cancel reservation AAAAAA."}],
        tools=[RESPOND_TOOL],
        tool_choice="required",
        sandbox_config={"runtime": "tau2"},
        sandbox_provider=provider,
    )

    assert model.calls == 1
    assert result["loop_exit_reason"] is None
    assert provider.sandbox.calls[0][0] == "respond"
    wrapped_message = provider.sandbox.calls[0][1]["message"]
    assert "</think>" not in wrapped_message
    assert "reason for cancellation" in wrapped_message.lower()
    assert wrapped_message == result["answer"]
    assert not any(step["status"] == "retry_required_tool_call" for step in result["agent_trace"])


@pytest.mark.fast
def test_real_0421_harmony_xml_response_reaches_tool_router_without_retry() -> None:
    text = _fixture_text("0421_tau_harmony_xml_respond_response.txt")
    model = _StaticTextModel(text)
    provider = _Provider()
    loop = AgentLoop(
        backend=ModelBackend({"backend": model}),
        tool_router=ToolRouter(),
        max_turns=200,
        tool_call_retry_budget=3,
    )

    result = loop.run(
        messages=[{"role": "user", "content": "My mobile data is not working."}],
        tools=[RESPOND_TOOL],
        tool_choice="required",
        sandbox_config={"runtime": "tau2"},
        sandbox_provider=provider,
    )

    assert model.calls == 1
    assert provider.sandbox.calls[0][0] == "respond"
    assert provider.sandbox.calls[0][1]["message"] == result["answer"]
    assert result["loop_exit_reason"] is None
    assert result["observability_events"] == []
    assert result["agent_trace"][0]["trace_role"] == "tool"
    assert result["agent_trace"][0]["name"] == "respond"
    assert result["agent_trace"][1]["status"] == "success"
    assert not any(step["status"] == "retry_required_tool_call" for step in result["agent_trace"])


@pytest.mark.fast
def test_real_0421_plain_text_response_exhausts_retry_budget_not_max_turns() -> None:
    text = _fixture_text("0421_tau_plain_text_missing_tool_response.txt")
    backend = _StaticTextModel(text)
    loop = AgentLoop(
        backend=backend,
        tool_router=ToolRouter(),
        max_turns=200,
        tool_call_retry_budget=3,
    )

    result = loop.run(
        messages=[{"role": "user", "content": "My mobile data is not working."}],
        tools=[RESPOND_TOOL],
        tool_choice="required",
    )

    assert backend.calls == 3
    assert result["answer"] == ""
    assert result["loop_exit_reason"] == "tool_call_retry_budget"
    assert len(result["agent_trace"]) == 3
    assert [step["turn_index"] for step in result["agent_trace"]] == [1, 2, 3]
    assert [step["output"]["retry_count"] for step in result["agent_trace"]] == [1, 2, 3]
    assert all(step["status"] == "retry_required_tool_call" for step in result["agent_trace"])
    assert all(payload["tool_choice"] == "required" for payload in backend.payloads)
    events = result["observability_events"]
    retry_events = [event for event in events if event["event"] == "agent_retry_missing_tool_call"]
    exhausted_events = [event for event in events if event["event"] == "agent_loop_exhausted"]
    assert len(retry_events) == 3
    assert len(exhausted_events) == 1
    assert exhausted_events[0]["payload"]["reason"] == "tool_call_retry_budget"
    assert exhausted_events[0]["payload"]["turn_index"] == 3
    assert exhausted_events[0]["payload"]["budget"] == 3
    assert not any(event["payload"].get("reason") == "max_turns" for event in exhausted_events)


@pytest.mark.fast
def test_real_0421_fixture_documents_original_too_many_errors_artifact() -> None:
    cost_payload = json.loads(_fixture_text("0421_tau_old_failed_cost.json"))

    assert cost_payload["termination_reason"] == "too_many_errors"
    assert cost_payload["agent_cost"] > 1_000_000


@pytest.mark.fast
def test_real_0421_tau2_fixture_documents_user_side_tool_error() -> None:
    trace_step = json.loads(_fixture_text("0421_tau2_unknown_user_side_tool_error.json"))

    assert trace_step["name"] == "check_network_status"
    assert trace_step["status"] == "failed"
    assert trace_step["output"]["content"] == "Error: Tool 'check_network_status' not found."


@pytest.mark.fast
def test_real_0421_terminal_signals_parse_from_qwen_user_simulator_outputs() -> None:
    payload = _fixture_json("0421_qwen_real_terminal_signals.json")

    for item in payload["messages"]:
        signal = parse_terminal_signal(item["content"])

        assert signal is not None
        assert signal.kind == item["kind"]
        assert signal.canonical == CANONICAL[item["kind"]]
        assert signal.raw == item["raw"]


@pytest.mark.fast
def test_real_0421_simulation_terminated_failed_tool_result_stops_loop() -> None:
    trace_step = _fixture_json("0421_qwen_simulation_terminated_tool_trace.json")
    backend = _StaticTextModel("")
    backend.invoke = lambda _payload: {
        "answer": "",
        "tool_calls": [
            {
                "id": "call_real_0421_terminated",
                "type": "function",
                "function": {
                    "name": trace_step["name"],
                    "arguments": trace_step["input"],
                },
            }
        ],
    }
    provider = _Provider()
    provider.sandbox.exec_tool = lambda name, arguments: dict(trace_step["output"])
    loop = AgentLoop(
        backend=backend,
        tool_router=ToolRouter(),
        max_turns=200,
        tool_call_retry_budget=3,
    )

    result = loop.run(
        messages=[{"role": "user", "content": "Show me product details."}],
        tools=[
            {
                "type": "function",
                "function": {"name": trace_step["name"], "parameters": {}},
                "x-gage": {"final_answer_from": "final_answer"},
            }
        ],
        tool_choice="required",
        sandbox_config={"runtime": "tau2"},
        sandbox_provider=provider,
    )

    assert result["answer"] == "simulation_terminated"
    assert result["loop_exit_reason"] is None
    assert result["agent_trace"][0]["status"] == "failed"
    assert result["agent_trace"][1]["output"] == {
        "answer": "simulation_terminated",
        "final_from_tool": trace_step["name"],
    }
