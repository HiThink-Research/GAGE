from __future__ import annotations

from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from gage_eval.agent_eval_kits.tau2.runtime import Tau2RuntimeEntry
from gage_eval.sandbox.protocols import (
    StateQueryProtocol,
    TaskInitProtocol,
    ToolExecutionProtocol,
)
from gage_eval.sandbox.tau2_runtime import (
    Tau2Runtime,
    _normalize_tau2_user_model_args,
    _resolve_agent_usage_cost,
    _resolve_tau2_user_simulator_runtime_config,
    _termination_reason,
)
from tests._support.stubs.tau2_stub import STOP, install_tau2_stub


def _build_sample(domain: str = "airline") -> dict:
    task = {
        "id": "task-1",
        "user_scenario": {"instructions": "Call support"},
        "evaluation_criteria": {"reward_basis": ["DB"]},
    }
    return {
        "id": "sample-1",
        "metadata": {"tau2": {"domain": domain, "trial": 0, "seed": 1}},
        "raw_assets": {"tau2": {"task": task}},
    }


def test_tau2_runtime_basic_flow(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    runtime = Tau2Runtime()
    runtime.start({"runtime_configs": {"data_dir": str(tmp_path)}})

    sample = _build_sample()
    init_output = runtime.initialize_task(sample)

    assert init_output["messages"]
    assert len(sample["messages"]) == 1  # synthetic greeting filtered; DUT sees only user_message

    respond_out = runtime.exec_tool("respond", {"message": "hello"})
    assert respond_out["user_message"] == "user_response"

    tool_out = runtime.exec_tool("lookup", {"query": "x"})
    assert "content" in tool_out


def test_tau2_runtime_respond_tool_schema_marks_final_answer(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    runtime = Tau2Runtime()
    runtime.start({"runtime_configs": {"data_dir": str(tmp_path)}})

    sample = _build_sample(domain="telecom")
    runtime.initialize_task(sample)

    respond_tool = next(
        tool
        for tool in sample["tools"]
        if tool.get("function", {}).get("name") == "respond"
    )
    assert respond_tool["x-gage"]["final_answer_from"] == "final_answer"


def test_tau2_runtime_gage_instruction_blocks_user_side_tool_hallucination(
    tmp_path: Path, monkeypatch
) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    runtime = Tau2Runtime()
    runtime.start({"runtime_configs": {"data_dir": str(tmp_path)}})

    sample = _build_sample(domain="telecom")
    runtime.initialize_task(sample)

    instruction = sample["metadata"]["tau2"]["gage_instruction"]
    assert "Only call tools listed in the provided tools schema" in instruction
    assert "user-side device tools" in instruction
    assert "check_status_bar" in instruction
    assert "Do not call them directly" in instruction
    assert "Before transferring to a human agent" in instruction
    assert "exhaust every troubleshooting step listed in the policy" in instruction
    assert "Before telling the user the problem is resolved" in instruction
    assert "can_send_mms" in instruction
    assert "connect_vpn" in instruction
    assert "disconnect_vpn" in instruction
    assert "run_speed_test" in instruction
    assert "final verification" in instruction


def test_tau2_runtime_unknown_user_side_tool_error_guides_agent(
    tmp_path: Path, monkeypatch
) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    runtime = Tau2Runtime()
    runtime.start({"runtime_configs": {"data_dir": str(tmp_path)}})
    runtime.initialize_task(_build_sample(domain="telecom"))

    env = runtime._env
    env.get_user_tools = lambda: [SimpleNamespace(name="check_network_status")]

    def fake_response(tool_call):
        return SimpleNamespace(
            id=tool_call.id,
            role="tool",
            content="Error: Tool 'check_network_status' not found.",
            requestor="assistant",
            error=True,
        )

    env.get_response = fake_response

    response = runtime.exec_tool("check_network_status", {})

    content = response["content"]
    assert "Tool 'check_network_status' is a user-side device tool" in content
    assert "Do not call it directly" in content
    assert "respond" in content
    assert "Available agent tools: lookup, respond" in content


def test_tau2_runtime_user_tools_and_stop(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path, force_user_tool_call=True)
    runtime = Tau2Runtime()
    runtime.start({"runtime_configs": {"data_dir": str(tmp_path)}})

    sample = _build_sample(domain="telecom")
    runtime.initialize_task(sample)
    respond_out = runtime.exec_tool("respond", {"message": "please stop"})

    assert respond_out["final_answer"] == STOP
    state = runtime.get_state()
    term = state["termination_reason"]
    assert (term.value if hasattr(term, "value") else str(term)) == "user_stop"


def test_tau2_runtime_get_state_tracks_termination_detail(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    runtime = Tau2Runtime()
    runtime.start({"runtime_configs": {"data_dir": str(tmp_path)}})

    sample = _build_sample()
    runtime.initialize_task(sample)

    assert runtime.get_state().get("termination_detail") is None


def test_tau2_runtime_initialize_task_resets_per_task_state(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    runtime = Tau2Runtime()
    runtime.start({"runtime_configs": {"data_dir": str(tmp_path)}})

    runtime.initialize_task(_build_sample())
    runtime.mark_agent_exhausted("no_tool_call_from_agent")

    runtime.initialize_task(_build_sample(domain="telecom"))
    state = runtime.get_state()
    assert state["termination_reason"] is None
    assert state["termination_detail"] is None
    assert len(state["messages"]) == 2

    response = runtime.exec_tool("respond", {"message": "hello"})
    assert response.get("final_answer") != "simulation_terminated"


def test_tau2_runtime_satisfies_protocols(tmp_path: Path, monkeypatch) -> None:
    """Tau2Runtime satisfies all three tool-protocol runtime contracts."""
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    runtime = Tau2Runtime()
    runtime.start({"runtime_configs": {"data_dir": str(tmp_path)}})

    assert isinstance(runtime, ToolExecutionProtocol)
    assert isinstance(runtime, StateQueryProtocol)
    assert isinstance(runtime, TaskInitProtocol)


@pytest.mark.parametrize("initial_reason", ["user_stop", "max_steps"])
def test_tau2_runtime_mark_agent_exhausted_no_override(initial_reason: str) -> None:
    runtime = Tau2Runtime()
    runtime._termination_reason = _termination_reason(initial_reason)
    runtime._termination_detail = "already_terminated"

    runtime.mark_agent_exhausted("tool_call_retry_budget")

    state = runtime.get_state()
    term = state["termination_reason"]
    assert (term.value if hasattr(term, "value") else str(term)) == initial_reason
    assert state["termination_detail"] == "already_terminated"


def test_tau2_runtime_mark_agent_exhausted_sets_agent_error_reason_and_detail() -> None:
    runtime = Tau2Runtime()

    runtime.mark_agent_exhausted("no_tool_call_from_agent")

    state = runtime.get_state()
    term = state["termination_reason"]
    assert (term.value if hasattr(term, "value") else str(term)) == "agent_error"
    assert state["termination_detail"] == "no_tool_call_from_agent"


def test_tau2_runtime_exec_reports_protocol_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    """exec() raises NotImplementedError with a clear protocol-mismatch message."""
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    runtime = Tau2Runtime()
    runtime.start({"runtime_configs": {"data_dir": str(tmp_path)}})

    with pytest.raises(NotImplementedError, match="exec_tool|tool protocol"):
        runtime.exec("ls")


def test_tau2_runtime_maps_legacy_error_reasons_with_new_enum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TerminationReason(str, Enum):
        USER_STOP = "user_stop"
        AGENT_STOP = "agent_stop"
        MAX_STEPS = "max_steps"
        TOO_MANY_ERRORS = "too_many_errors"

    monkeypatch.setattr(
        "gage_eval.sandbox.tau2_runtime.resolve_tau2_termination_reason",
        lambda reason, fallback="too_many_errors": (
            TerminationReason.TOO_MANY_ERRORS
            if reason in {"agent_error", "user_error", None}
            else getattr(TerminationReason, str(reason).upper())
        ),
    )

    assert _termination_reason("agent_error") == TerminationReason.TOO_MANY_ERRORS
    assert _termination_reason("user_error") == TerminationReason.TOO_MANY_ERRORS


def test_tau2_runtime_normalizes_ollama_chat_api_base() -> None:
    normalized = _normalize_tau2_user_model_args(
        model="ollama_chat/qwen3-vl:2b-instruct",
        model_args={"api_base": "http://127.0.0.1:11434/v1", "api_key": "dummy"},
    )

    assert normalized == {
        "api_base": "http://127.0.0.1:11434",
        "api_key": "dummy",
    }


def test_tau2_runtime_prefers_user_simulator_config_over_legacy_user_model() -> None:
    resolved = _resolve_tau2_user_simulator_runtime_config(
        {
            "user_model": "legacy-model",
            "user_model_args": {"temperature": 0.5},
            "user_simulator": {
                "model": "gpt-4.1",
                "model_args": {"temperature": 0.0},
            },
        }
    )

    assert resolved == {"model": "gpt-4.1", "model_args": {"temperature": 0.0}}


def test_tau2_runtime_prefers_kit_user_simulator_override() -> None:
    resolved = _resolve_tau2_user_simulator_runtime_config(
        {
            "user_model": "legacy-model",
            "user_model_args": {"temperature": 0.5},
        },
        override={
            "model": "openai/gpt-4.1",
            "model_args": {"api_base": "https://api.openai.com/v1"},
        },
    )

    assert resolved == {
        "model": "openai/gpt-4.1",
        "model_args": {
            "api_base": "https://api.openai.com/v1",
            "temperature": 0.0,
        },
    }


def test_tau2_kit_configures_user_simulator_before_initialize() -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.configured = None

        def configure_user_simulator(self, user_simulator) -> None:
            self.configured = user_simulator

        def initialize_task(self, _sample):
            return {"messages": [], "tools_schema": [], "metadata": {}}

    runtime = FakeRuntime()
    provider = SimpleNamespace(get_handle=lambda: SimpleNamespace(sandbox=runtime))
    session = SimpleNamespace(
        benchmark_config={
            "user_simulator": {
                "model": "gpt-4.1",
                "model_args": {"temperature": 0.0},
            }
        }
    )

    Tau2RuntimeEntry().bootstrap(
        session=session,
        sample={"id": "sample-1"},
        payload={},
        sandbox_provider=provider,
    )

    assert runtime.configured == {
        "model": "gpt-4.1",
        "model_args": {"temperature": 0.0},
    }


def test_tau2_runtime_records_agent_usage_from_total_tokens(
    tmp_path: Path,
    monkeypatch,
) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    runtime = Tau2Runtime()
    runtime.start({"runtime_configs": {"data_dir": str(tmp_path)}})

    runtime.record_agent_usage({"total_tokens": 42})

    assert runtime.get_state()["agent_cost"] == 42.0


def test_resolve_agent_usage_cost_prefers_total_tokens() -> None:
    assert _resolve_agent_usage_cost({"total_tokens": 12}) == 12.0
    assert _resolve_agent_usage_cost({"prompt_tokens": 4, "completion_tokens": 6}) == 10.0
    assert _resolve_agent_usage_cost({"input_tokens": 3, "output_tokens": 7}) == 10.0
    assert _resolve_agent_usage_cost({"cost_usd": 0.25, "total_tokens": 99}) == 0.25


def test_tau2_runtime_resolves_openai_http_user_simulator() -> None:
    resolved = _resolve_tau2_user_simulator_runtime_config(
        {},
        override={
            "type": "openai_http",
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "base_url": "http://localhost:8000/v1",
            "model_args": {"temperature": 0.0, "api_key": "dummy"},
        },
    )

    assert resolved["model"] == "openai/Qwen/Qwen2.5-72B-Instruct"
    assert resolved["model_args"]["api_base"] == "http://localhost:8000/v1"
    assert resolved["model_args"]["api_key"] == "dummy"
    assert resolved["model_args"]["temperature"] == 0.0


def test_tau2_runtime_openai_http_preserves_existing_openai_prefix() -> None:
    resolved = _resolve_tau2_user_simulator_runtime_config(
        {},
        override={
            "type": "openai_http",
            "model": "openai/my-model",
            "model_args": {"api_base": "http://localhost:8000/v1"},
        },
    )

    assert resolved["model"] == "openai/my-model"


def test_tau2_runtime_openai_http_reads_api_base_from_model_args() -> None:
    resolved = _resolve_tau2_user_simulator_runtime_config(
        {},
        override={
            "type": "openai_http",
            "model": "my-model",
            "model_args": {"api_base": "http://vllm-host:8000/v1"},
        },
    )

    assert resolved["model_args"]["api_base"] == "http://vllm-host:8000/v1"


def test_tau2_runtime_openai_http_reads_base_url_from_model_args() -> None:
    resolved = _resolve_tau2_user_simulator_runtime_config(
        {},
        override={
            "type": "openai_http",
            "model": "my-model",
            "model_args": {"base_url": "http://vllm-host:8000/v1"},
        },
    )

    assert resolved["model_args"]["api_base"] == "http://vllm-host:8000/v1"
    assert "base_url" not in resolved["model_args"]


def test_tau2_runtime_openai_http_does_not_strip_v1_from_api_base() -> None:
    normalized = _normalize_tau2_user_model_args(
        model="openai/Qwen2.5-72B-Instruct",
        model_args={"api_base": "http://localhost:8000/v1"},
    )

    assert normalized["api_base"] == "http://localhost:8000/v1"


def test_tau2_runtime_openai_http_applies_env_base_url(monkeypatch) -> None:
    monkeypatch.setenv("TAU2_USER_BASE_URL", "http://env-host:9000/v1")
    monkeypatch.setenv("TAU2_USER_MODEL", "my-env-model")

    resolved = _resolve_tau2_user_simulator_runtime_config(
        {},
        override={"type": "openai_http"},
    )

    assert resolved["model"] == "openai/my-env-model"
    assert resolved["model_args"]["api_base"] == "http://env-host:9000/v1"


def test_tau2_runtime_openai_http_preserves_tool_choice_auto_override() -> None:
    # tool_choice in model_args gets passed as **llm_args to tau2's generate(),
    # which has tool_choice as a named parameter and keeps OpenAI-compatible
    # tool calling explicit for user-simulator endpoints.
    resolved = _resolve_tau2_user_simulator_runtime_config(
        {},
        override={
            "type": "openai_http",
            "model": "my-model",
            "model_args": {
                "api_base": "http://localhost:8000/v1",
                "tool_choice": "auto",
            },
        },
    )

    assert resolved["model_args"]["tool_choice"] == "auto"
