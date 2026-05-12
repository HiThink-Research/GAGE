from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gage_eval.agent_eval_kits.tau2.tools import build_tool_registry as build_tau2_tool_registry
from gage_eval.agent_eval_kits.tau2.artifacts import persist_tau2_artifacts
from gage_eval.agent_eval_kits.tau2.config_schema import (
    Tau2KitConfig,
    normalize_user_simulator_config,
)
from gage_eval.agent_eval_kits.tau2.sub_workflows.framework_loop import _finalize_loop_result
from gage_eval.agent_eval_kits.tau2.local_runtime import (
    CANONICAL,
    Tau2Runtime,
    TerminalSignal,
    _normalize_user_message_tool_calls,
    _normalize_user_tool_name,
    parse_terminal_signal,
)
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
from gage_eval.agent_runtime.resources.contracts import ResourceLease
from gage_eval.agent_runtime.schedulers.framework_loop import FrameworkLoopScheduler
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.agent_runtime.tooling.contracts import ToolSchemaIR
from gage_eval.agent_runtime.tooling.provider_adapters import Tau2ToolDialectParser
from gage_eval.agent_runtime.tooling.registry import RuntimeToolRegistry
from gage_eval.agent_runtime.tooling.router import ToolRouter
from tests._support.stubs.tau2_stub import install_tau2_stub


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "tau2" / "runtime_samples"


def _sample(domain: str = "airline") -> dict:
    return {
        "id": "tau2-sample-1",
        "metadata": {"tau2": {"domain": domain, "trial": 0, "seed": 1}},
        "raw_assets": {
            "tau2": {
                "task": {
                    "id": "task-1",
                    "user_scenario": {"instructions": "Need support"},
                    "evaluation_criteria": {"reward_basis": ["DB"]},
                }
            }
        },
    }


class _RuntimeForArtifacts:
    def __init__(self, state: dict[str, object]) -> None:
        self.state = state
        self.mark_calls: list[str] = []
        self.recorded_usage: object = None

    def get_state(self) -> dict[str, object]:
        return dict(self.state)

    def mark_agent_exhausted(self, detail: str) -> None:
        self.mark_calls.append(detail)
        if self.state.get("termination_reason") is None:
            self.state["termination_reason"] = "agent_error"
            self.state["termination_detail"] = detail

    def record_agent_usage(self, usage: object) -> None:
        self.recorded_usage = usage
        if isinstance(usage, dict):
            self.state["agent_cost"] = usage.get("cost_usd")
            self.state["agent_total_tokens"] = usage.get("total_tokens")


class _Provider:
    def __init__(self, runtime: object) -> None:
        self.runtime = runtime

    def get_handle(self) -> SimpleNamespace:
        return SimpleNamespace(sandbox=self.runtime)


def _session(tmp_path: Path) -> SimpleNamespace:
    sample_root = tmp_path / "sample"
    artifacts_dir = sample_root / "artifacts"
    artifacts_dir.mkdir(parents=True)
    return SimpleNamespace(
        artifact_layout={
            "sample_root": str(sample_root),
            "artifacts_dir": str(artifacts_dir),
        }
    )


def _fixture_text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def _fixture_json(name: str):
    return json.loads(_fixture_text(name))


class _StaticResponseBackend:
    def __init__(self, response):
        self.response = response
        self.payloads: list[dict] = []

    async def ainvoke(self, payload: dict) -> dict:
        self.payloads.append(dict(payload))
        return self.response() if callable(self.response) else self.response


class _Tau2RespondLease:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def exec_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        if name == "respond":
            return {"final_answer": arguments.get("message"), "user_message": arguments.get("message")}
        return {"ok": True}

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self.exec_tool(name, arguments)


def _framework_session() -> AgentRuntimeSession:
    return AgentRuntimeSession(
        session_id="session-1",
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        benchmark_kit_id="tau2",
        scheduler_type="framework_loop",
        resource_lease=ResourceLease(
            lease_id="lease-1",
            resource_kind="local_process",
            profile_id="tau2",
            lifecycle="per_sample",
        ),
    )


def _framework_bundle() -> SchedulerWorkflowBundle:
    return SchedulerWorkflowBundle(
        bundle_id="tau2.framework_loop",
        benchmark_kit_id="tau2",
        scheduler_type="framework_loop",
        build_loop_inputs=lambda **kwargs: {
            "required_tool": None if (kwargs.get("payload") or {}).get("tool_choice") == "none" else "respond",
            "plain_text_response_tool": "respond",
            "plain_text_response_argument": "message",
            "refresh_tool_schemas": True,
            "tool_text_parser": "tau2",
            "tool_result_user_message_field": "user_message",
        },
    )


def _run_tau2_scheduler(*, backend, registry: RuntimeToolRegistry, lease=None, max_turns: int = 3, payload=None):
    scheduler = FrameworkLoopScheduler(
        backend=backend,
        tool_router=ToolRouter(registry),
        tool_registry=registry,
        max_turns=max_turns,
    )
    return asyncio.run(
        scheduler.arun(
            session=_framework_session(),
            sample={"messages": [{"role": "user", "content": "hi"}]},
            payload={"environment_lease": lease or _Tau2RespondLease(), **(payload or {})},
            workflow_bundle=_framework_bundle(),
            sandbox_provider=None,
        )
    )


def test_real_0421_harmony_xml_response_parses_as_respond_tool_call() -> None:
    calls = Tau2ToolDialectParser().parse(
        _fixture_text("0421_tau_harmony_xml_respond_response.txt"),
        dialect="auto",
        turn_index=1,
    )

    assert len(calls) == 1
    assert calls[0].name == "respond"
    assert "phone number" in calls[0].arguments()["message"].lower()


def test_real_0421_think_tail_bare_json_response_parses_as_respond_tool_call() -> None:
    calls = Tau2ToolDialectParser().parse(
        _fixture_text("0421_tau2_think_tail_bare_json_respond_response.txt"),
        dialect="auto",
        turn_index=1,
    )

    assert len(calls) == 1
    assert calls[0].name == "respond"
    assert "verify your account" in calls[0].arguments()["message"].lower()


def test_real_0421_gemma4_bare_call_preserves_full_message() -> None:
    calls = Tau2ToolDialectParser().parse(
        _fixture_text("0421_gemma4_airline_bare_call_respond_response.txt"),
        dialect="gemma",
        turn_index=1,
    )

    assert len(calls) == 1
    message = calls[0].arguments()["message"]
    assert "reservation EHGLP3" in message
    assert "reason for the cancellation" in message


def test_real_0421_harmony_xml_response_reaches_tool_router_without_retry() -> None:
    text = _fixture_text("0421_tau_harmony_xml_respond_response.txt")
    lease = _Tau2RespondLease()
    result = _run_tau2_scheduler(
        backend=_StaticResponseBackend({"answer": text}),
        registry=build_tau2_tool_registry(),
        lease=lease,
        max_turns=2,
    )

    assert result.status == "completed"
    assert lease.calls[0][0] == "respond"
    assert "phone number" in lease.calls[0][1]["message"].lower()
    assert not any(step["name"] == "missing_required_tool_call" for step in result.agent_output["agent_trace"])


def test_real_0421_qwen_plain_text_response_reaches_respond_with_think_tail_stripped() -> None:
    text = _fixture_text("0421_qwen_gpt_airline_plain_text_response.txt")
    lease = _Tau2RespondLease()
    result = _run_tau2_scheduler(
        backend=_StaticResponseBackend({"answer": text}),
        registry=build_tau2_tool_registry(),
        lease=lease,
        max_turns=2,
    )

    assert result.status == "completed"
    assert lease.calls[0][0] == "respond"
    message = lease.calls[0][1]["message"]
    assert "</think>" not in message
    assert "reason for cancellation" in message.lower()


def test_real_0421_simulation_terminated_failed_tool_result_stops_loop() -> None:
    trace_step = _fixture_json("0421_qwen_simulation_terminated_tool_trace.json")
    registry = RuntimeToolRegistry()
    registry.register_environment_tool(
        ToolSchemaIR(
            name=trace_step["name"],
            description="fixture tool",
            input_schema={"type": "object", "properties": {"product_id": {"type": "integer"}}},
            raw_schema={"type": "function", "function": {"name": trace_step["name"], "parameters": {}}},
        )
    )
    lease = _Tau2RespondLease()
    lease.exec_tool = lambda name, arguments: dict(trace_step["output"])
    result = _run_tau2_scheduler(
        backend=_StaticResponseBackend(
            {
                "tool_calls": [
                    {
                        "id": "call-real-0421",
                        "type": "function",
                        "function": {"name": trace_step["name"], "arguments": trace_step["input"]},
                    }
                ]
            }
        ),
        registry=registry,
        lease=lease,
        max_turns=2,
        payload={"required_tool": None},
    )

    assert result.status == "completed"
    assert result.agent_output["answer"] == "simulation_terminated"


def test_real_0421_terminal_signals_parse_from_user_simulator_outputs() -> None:
    payload = _fixture_json("0421_qwen_real_terminal_signals.json")

    for item in payload["messages"]:
        signal = parse_terminal_signal(item["content"])
        assert signal is not None
        assert signal.kind == item["kind"]
        assert signal.canonical == CANONICAL[item["kind"]]
        assert signal.raw == item["raw"]


def test_real_0421_fixture_documents_user_side_tool_error() -> None:
    trace_step = _fixture_json("0421_tau2_unknown_user_side_tool_error.json")

    assert trace_step["name"] == "check_network_status"
    assert trace_step["status"] == "failed"
    assert "not found" in trace_step["output"]["content"]


def test_tau2_default_greeting_stays_in_trajectory_but_not_agent_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    runtime = Tau2Runtime()
    runtime.start({"data_dir": str(tmp_path)})
    sample = _sample()

    init_output = runtime.initialize_task(sample)

    assert [message["role"] for message in init_output["messages"]] == ["user"]
    assert "Hi! How can I help you today?" not in json.dumps(init_output["messages"])
    assert init_output["metadata"]["tau2"]["gage_instruction"] == ""
    assert init_output["metadata"]["tau2"]["gemma4_tool_instruction"] == ""
    assert len(runtime.get_state()["messages"]) == 2


def test_tau2_parse_terminal_signal_variants() -> None:
    assert parse_terminal_signal("### out of scope ###") == TerminalSignal(
        kind="OUT_OF_SCOPE",
        canonical=CANONICAL["OUT_OF_SCOPE"],
        raw="### out of scope ###",
    )
    assert parse_terminal_signal("Thanks.\n### STOP ###") == TerminalSignal(
        kind="STOP",
        canonical=CANONICAL["STOP"],
        raw="### STOP ###",
    )
    assert parse_terminal_signal("### transfer ###").canonical == CANONICAL["TRANSFER"]
    assert parse_terminal_signal("### transfer ###", allowed=frozenset({"STOP"})) is None


def test_tau2_normalize_user_message_tool_calls() -> None:
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(name="check_roaming_status<|channel|>analysis"),
            SimpleNamespace(name="get_status_bar"),
        ]
    )

    normalized = _normalize_user_message_tool_calls(message)

    assert [call.name for call in normalized.tool_calls] == [
        "check_network_status",
        "check_status_bar",
    ]
    assert [call.name for call in message.tool_calls] == [
        "check_roaming_status<|channel|>analysis",
        "get_status_bar",
    ]


def test_tau2_tool_namespace_helpers_match_tau2patch() -> None:
    assert _normalize_user_tool_name("check_data_saver_mode") == "check_data_restriction_status"
    assert _normalize_user_tool_name("get_status_bar<|channel|>commentary") == "check_status_bar"
    assert _normalize_user_tool_name("run_speed_test") == "run_speed_test"


def test_tau2_openai_http_user_simulator_config_normalization() -> None:
    normalized = normalize_user_simulator_config(
        {
            "type": "openai_http",
            "model": "qwen3-8b",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "dummy",
            "temperature": 0.2,
        }
    )

    assert normalized == {
        "model": "qwen3-8b",
        "model_args": {
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "dummy",
            "temperature": 0.2,
        },
    }
    assert Tau2KitConfig.model_validate({"domain": "telecom", "user_simulator": normalized}).user_simulator == normalized


@pytest.mark.parametrize(
    ("loop_exit_reason", "expected_detail"),
    [
        ("tool_call_retry_budget", "no_tool_call_from_agent"),
        ("max_turns", "agent_loop_max_turns"),
    ],
)
def test_tau2_mark_loop_termination_writes_agent_exhausted_and_terminal_detail(
    tmp_path: Path,
    loop_exit_reason: str,
    expected_detail: str,
) -> None:
    runtime = _RuntimeForArtifacts(
        {
            "messages": [],
            "termination_reason": None,
            "termination_detail": None,
            "agent_cost": 0.0,
            "user_cost": 0.0,
        }
    )

    output = _finalize_loop_result(
        session=_session(tmp_path),
        sample={},
        scheduler_output={"loop_exit_reason": loop_exit_reason},
        sandbox_provider=_Provider(runtime),
    )

    assert runtime.mark_calls == [expected_detail]
    assert output["runtime_state"]["termination_detail"] == expected_detail
    state_payload = json.loads((tmp_path / "sample" / "artifacts" / "tau2_state.json").read_text())
    assert state_payload["agent_exhausted"] is True
    assert state_payload["termination_detail"] == expected_detail
    assert output["artifact_paths"]["tau2_cost"] == "artifacts/tau2_cost.json"


def test_tau2_trajectory_source_prefers_runtime_messages_over_agent_trace(tmp_path: Path) -> None:
    runtime = _RuntimeForArtifacts(
        {
            "messages": [{"role": "assistant", "content": "runtime message"}],
            "termination_reason": "agent_stop",
            "agent_cost": 0.0,
            "user_cost": 0.0,
        }
    )

    persist_tau2_artifacts(
        session=_session(tmp_path),
        scheduler_output={"agent_trace": [{"name": "stale_trace"}]},
        sandbox_provider=_Provider(runtime),
    )

    trajectory = json.loads((tmp_path / "sample" / "artifacts" / "tau2_trajectory.json").read_text())
    assert trajectory["source"] == "runtime_state.messages"
    assert trajectory["events"] == [{"role": "assistant", "content": "runtime message"}]


def test_tau2_record_agent_usage_preserved_in_summary(tmp_path: Path) -> None:
    runtime = _RuntimeForArtifacts(
        {
            "messages": [],
            "termination_reason": "agent_stop",
            "agent_cost": None,
            "user_cost": 0.0,
        }
    )

    _finalize_loop_result(
        session=_session(tmp_path),
        sample={},
        scheduler_output={"usage": {"total_tokens": 42, "cost_usd": 0.125}},
        sandbox_provider=_Provider(runtime),
    )

    assert runtime.recorded_usage == {"total_tokens": 42, "cost_usd": 0.125}
    state_payload = json.loads((tmp_path / "sample" / "artifacts" / "tau2_state.json").read_text())
    cost_payload = json.loads((tmp_path / "sample" / "artifacts" / "tau2_cost.json").read_text())
    assert state_payload["agent_cost"] == 0.125
    assert state_payload["agent_total_tokens"] == 42.0
    assert cost_payload["agent_cost"] == 0.125
    assert cost_payload["agent_total_tokens"] == 42.0
