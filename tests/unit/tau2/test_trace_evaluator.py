from __future__ import annotations

from types import SimpleNamespace

import pytest

from gage_eval.agent_eval_kits.tau2.judge.adapters import Tau2VerifierAdapter
from gage_eval.agent_eval_kits.tau2.judge.bridges import build_tau2_verifier_request
from gage_eval.agent_eval_kits.tau2.judge.executor import Tau2ExecutionRequest, execute_tau2_verifier
from gage_eval.agent_eval_kits.tau2.trace_mapping import evaluate_tau2_trace_order
from gage_eval.agent_runtime.verifier.contracts import VerifierInput
from tests._support.stubs.tau2_stub import install_tau2_stub


class _Trace:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict, str | None]] = []

    def emit(self, event: str, payload: dict, sample_id: str | None = None) -> None:
        self.events.append((event, payload, sample_id))


class _Runtime:
    def __init__(self) -> None:
        self.state = {
            "task_id": "task-1",
            "domain": "airline",
            "messages": [],
            "termination_reason": "agent_stop",
            "agent_cost": 0.0,
            "user_cost": 0.0,
        }

    def get_state(self) -> dict[str, object]:
        return dict(self.state)


class _Provider:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime

    def get_handle(self) -> SimpleNamespace:
        return SimpleNamespace(sandbox=self.runtime)


def _sample() -> dict:
    return {
        "id": "sample-1",
        "metadata": {"tau2": {"domain": "airline", "trial": 0, "seed": 1}},
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


def test_tau2_trace_evaluator_reconstructs_turn_and_tool_order() -> None:
    result = evaluate_tau2_trace_order(
        [
            {"event_type": "model.request", "actor": "scheduler", "payload": {"turn_index": 1}},
            {"event_type": "model.response", "actor": "scheduler", "payload": {"turn_index": 1}},
            {"event_type": "tool.call.normalized", "actor": "agent", "payload": {"turn_index": 1}},
            {"event_type": "tool.result", "actor": "runtime", "payload": {"tool_call_id": "call-1"}},
            {"event_type": "tool.result.injected", "actor": "scheduler", "payload": {"tool_call_id": "call-1"}},
        ]
    )

    assert result["valid"] is True
    assert result["turns"] == [
        {
            "turn_index": 1,
            "events": [
                "model.request",
                "model.response",
                "tool.call.normalized",
                "tool.result",
                "tool.result.injected",
            ],
        }
    ]


def test_tau2_verifier_input_carries_four_required_refs() -> None:
    request = build_tau2_verifier_request(
        sample_id="sample-1",
        sample=_sample(),
        scheduler_result={
            "artifact_paths": {
                "tau2_trajectory": "artifacts/tau2_trajectory.json",
                "tau2_state": "artifacts/tau2_state.json",
                "trace": "traces/sample-1.jsonl",
            },
            "runtime_state": {"messages": []},
        },
        runtime_context={
            "trace_events": [
                {"event_type": "model.request", "actor": "scheduler", "payload": {"turn_index": 1}},
                {"event_type": "model.response", "actor": "scheduler", "payload": {"turn_index": 1}},
                {"event_type": "tool.call.normalized", "actor": "agent", "payload": {"turn_index": 1}},
                {"event_type": "tool.result", "actor": "runtime", "payload": {"tool_call_id": "call-1"}},
                {"event_type": "tool.result.injected", "actor": "scheduler", "payload": {"tool_call_id": "call-1"}},
            ]
        },
    )

    assert request["trajectory_ref"] == "artifacts/tau2_trajectory.json"
    assert request["runtime_state_ref"] == "artifacts/tau2_state.json"
    assert request["trace_ref"] == "traces/sample-1.jsonl"
    assert request["tool_trace_summary"]["turn_count"] == 1
    assert request["tool_trace_summary"]["turns"][0]["events"] == [
        "model.request",
        "model.response",
        "tool.call.normalized",
        "tool.result",
        "tool.result.injected",
    ]


def test_tau2_verifier_request_falls_back_to_runtime_state_when_scheduler_state_is_bootstrap_only() -> None:
    runtime = _Runtime()
    runtime.state["messages"] = [{"role": "assistant", "content": "final runtime message"}]

    request = build_tau2_verifier_request(
        sample_id="sample-1",
        sample=_sample(),
        scheduler_result={
            "runtime_state": {"initialize_result": {"messages": []}},
            "artifact_paths": {},
        },
        runtime_context={"sandbox_provider": _Provider(runtime)},
    )

    assert request["runtime_state"]["messages"] == [
        {"role": "assistant", "content": "final runtime message"}
    ]


def test_tau2_verifier_timeout_maps_failure_code() -> None:
    result = execute_tau2_verifier(
        Tau2ExecutionRequest(
            sample_id="sample-1",
            sample=_sample(),
            runtime_state={},
            scheduler_result={},
            timeout_s=1,
        ),
        evaluator=lambda _request: (_ for _ in ()).throw(TimeoutError("slow evaluator")),
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "verifier.executor.timeout"
    assert result["failure_reason"] == "verifier_timeout"


def test_tau2_verifier_evaluator_exception_maps_failure_code() -> None:
    result = execute_tau2_verifier(
        Tau2ExecutionRequest(
            sample_id="sample-1",
            sample=_sample(),
            runtime_state={},
            scheduler_result={},
            timeout_s=1,
        ),
        evaluator=lambda _request: (_ for _ in ()).throw(
            ValueError("trajectory tool output mismatch")
        ),
    )

    assert result["status"] == "failed"
    assert result["resolved"] is False
    assert result["score"] == 0.0
    assert result["failure_code"] == "verifier.executor.failed"
    assert result["failure_reason"] == "ValueError: trajectory tool output mismatch"
    assert result["summary"] == "Tau2 verifier crashed during evaluation"


def test_tau2_adapter_runs_kit_owned_verifier_and_emits_result_trace(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    trace = _Trace()
    runtime = _Runtime()

    result = Tau2VerifierAdapter().run(
        VerifierInput(
            benchmark_kit_id="tau2",
            scheduler_type="framework_loop",
            sample_id="sample-1",
            sample=_sample(),
            scheduler_result={"artifact_paths": {}, "runtime_state": runtime.get_state()},
            runtime_context={
                "sandbox_provider": _Provider(runtime),
                "trace": trace,
                "trace_events": [],
            },
            verifier_resources={},
        )
    )

    assert result.status == "completed"
    assert result.payload["tau2"]["reward"] == 1.0
    event = next(item for item in trace.events if item[0] == "verifier.result")
    assert event[1]["metric"]["score"] == 1.0
    assert event[1]["verifier_result"]["tau2"]["reward"] == 1.0
