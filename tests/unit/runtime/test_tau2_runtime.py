from __future__ import annotations

from pathlib import Path
from enum import Enum
from typing import Any, Dict, Protocol, runtime_checkable

import pytest

from gage_eval.agent_eval_kits.tau2.local_runtime import (
    Tau2Runtime,
    _normalize_tau2_user_model_args,
    _resolve_tau2_user_simulator_runtime_config,
    _resolve_agent_usage_cost,
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
    runtime.start({"data_dir": str(tmp_path)})

    sample = _build_sample()
    init_output = runtime.initialize_task(sample)

    assert init_output["messages"]
    assert len(sample["messages"]) == 1
    assert sample["messages"][0]["role"] == "user"
    assert len(runtime.get_state()["messages"]) == 2

    respond_out = runtime.exec_tool("respond", {"message": "hello"})
    assert respond_out["user_message"] == "user_response"

    tool_out = runtime.exec_tool("lookup", {"query": "x"})
    assert "content" in tool_out


def test_tau2_runtime_user_tools_and_stop(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path, force_user_tool_call=True)
    runtime = Tau2Runtime()
    runtime.start({"data_dir": str(tmp_path)})

    sample = _build_sample(domain="telecom")
    runtime.initialize_task(sample)
    respond_out = runtime.exec_tool("respond", {"message": "please stop"})

    assert respond_out["final_answer"] == STOP
    state = runtime.get_state()
    term = state["termination_reason"]
    assert (term.value if hasattr(term, "value") else str(term)) == "user_stop"


def test_tau2_runtime_satisfies_protocols(tmp_path: Path, monkeypatch) -> None:
    """Tau2Runtime satisfies all three tool-protocol runtime contracts."""
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    runtime = Tau2Runtime()
    runtime.start({"data_dir": str(tmp_path)})

    assert isinstance(runtime, ToolExecutionProtocol)
    assert isinstance(runtime, StateQueryProtocol)
    assert isinstance(runtime, TaskInitProtocol)


def test_tau2_runtime_exec_reports_protocol_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    """exec() raises NotImplementedError with a clear protocol-mismatch message."""
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    runtime = Tau2Runtime()
    runtime.start({"data_dir": str(tmp_path)})

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
        "gage_eval.agent_eval_kits.tau2.local_runtime.resolve_tau2_termination_reason",
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


def test_tau2_runtime_injects_litellm_provider_for_unprefixed_local_qwen_model() -> None:
    resolved = _resolve_tau2_user_simulator_runtime_config(
        {
            "_backend_config": {"provider": "openai"},
            "user_simulator": {
                "model": "qwen/qwen3.5-9b",
                "model_args": {
                    "api_base": "http://127.0.0.1:1234/v1",
                    "api_key": "dummy",
                },
            },
        }
    )

    assert resolved["model_args"]["custom_llm_provider"] == "openai"


def test_tau2_runtime_does_not_override_existing_litellm_provider() -> None:
    resolved = _resolve_tau2_user_simulator_runtime_config(
        {
            "_backend_config": {"provider": "openai"},
            "user_simulator": {
                "model": "qwen/qwen3.5-9b",
                "model_args": {
                    "custom_llm_provider": "lm_studio",
                    "api_base": "http://127.0.0.1:1234/v1",
                },
            },
        }
    )

    assert resolved["model_args"]["custom_llm_provider"] == "lm_studio"


def test_tau2_runtime_records_agent_usage_cost_and_tokens_separately(
    tmp_path: Path,
    monkeypatch,
) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    runtime = Tau2Runtime()
    runtime.start({"data_dir": str(tmp_path)})

    runtime.record_agent_usage({"total_tokens": 42, "cost_usd": 0.125})

    state = runtime.get_state()
    assert state["agent_cost"] == 0.125
    assert state["agent_total_tokens"] == 42.0


def test_resolve_agent_usage_cost_uses_usd_not_token_counts() -> None:
    assert _resolve_agent_usage_cost({"total_tokens": 12}) is None
    assert _resolve_agent_usage_cost({"prompt_tokens": 4, "completion_tokens": 6}) is None
    assert _resolve_agent_usage_cost({"input_tokens": 3, "output_tokens": 7}) is None
    assert _resolve_agent_usage_cost({"cost_usd": 0.25, "total_tokens": 99}) == 0.25
@runtime_checkable
class ToolExecutionProtocol(Protocol):
    def exec_tool(self, name: str, arguments: Any) -> Dict[str, Any]:
        ...


@runtime_checkable
class StateQueryProtocol(Protocol):
    def get_state(self) -> Dict[str, Any]:
        ...


@runtime_checkable
class TaskInitProtocol(Protocol):
    def initialize_task(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        ...
