from __future__ import annotations

from types import SimpleNamespace

from gage_eval.agent_runtime.executor import _normalize_judge_output
from gage_eval.agent_runtime.contracts.scheduler import SchedulerResult
from gage_eval.agent_runtime.verifier.adapters import NativeVerifierAdapter
from gage_eval.agent_runtime.verifier.contracts import VerifierInput


def test_native_verifier_requires_expected_answer_for_terminal_bench() -> None:
    adapter = NativeVerifierAdapter("terminal_bench.native_verifier")
    verifier_input = VerifierInput(
        benchmark_kit_id="terminal_bench",
        scheduler_type="installed_client",
        sample_id="terminal-1",
        sample={"id": "terminal-1"},
        scheduler_result={"agent_output": {"answer": "done"}},
        runtime_context={},
        verifier_resources={},
    )

    result = adapter.run(verifier_input)

    assert result.payload["resolved"] is False
    assert result.payload["failure_reason"] == "missing_expected_answer"
    assert result.payload["score"] == 0.0


def test_native_verifier_matches_expected_answer_for_terminal_bench() -> None:
    adapter = NativeVerifierAdapter("terminal_bench.native_verifier")
    verifier_input = VerifierInput(
        benchmark_kit_id="terminal_bench",
        scheduler_type="installed_client",
        sample_id="terminal-1",
        sample={"id": "terminal-1", "expected_answer": "done"},
        scheduler_result={"agent_output": {"answer": "done"}},
        runtime_context={},
        verifier_resources={},
    )

    result = adapter.run(verifier_input)

    assert result.payload["resolved"] is True
    assert result.payload["failure_reason"] is None
    assert result.payload["score"] == 1.0


def test_normalize_judge_output_uses_appworld_tgc_for_resolution() -> None:
    normalized = _normalize_judge_output(
        {
            "status": "completed",
            "diagnostic_reason": "verifier_assertion_failed",
            "diagnostic_details": {
                "verifier_failures": [{"label": "bad"}],
            },
            "appworld": {
                "tgc": 0.0,
                "sgc": 0.0,
                "tests": {"passes": [{"label": "ok"}], "fails": [{"label": "bad"}]},
            },
        },
        scheduler_result=None,
        judge_source="appworld.verifier_adapter.run",
    )

    assert normalized["resolved"] is False
    assert normalized["score"] == 0.0
    assert normalized["failure_reason"] == "task_incomplete"
    assert normalized["diagnostic_reason"] == "verifier_assertion_failed"
    assert normalized["diagnostic_details"]["verifier_failures"][0]["label"] == "bad"


def test_normalize_judge_output_uses_tau2_reward_for_resolution() -> None:
    normalized = _normalize_judge_output(
        {
            "status": "completed",
            "diagnostic_reason": "too_many_errors",
            "diagnostic_details": {
                "tau2": {"termination_reason": "too_many_errors"},
            },
            "tau2": {
                "reward": 0.0,
                "termination_reason": "too_many_errors",
            },
        },
        scheduler_result=None,
        judge_source="tau2.verifier_adapter.run",
    )

    assert normalized["resolved"] is False
    assert normalized["score"] == 0.0
    assert normalized["failure_reason"] == "too_many_errors"
    assert normalized["diagnostic_reason"] == "too_many_errors"
    assert normalized["diagnostic_details"]["tau2"]["termination_reason"] == "too_many_errors"


def test_normalize_judge_output_surfaces_trace_summary_for_swebench_failures() -> None:
    scheduler_result = SchedulerResult(
        scheduler_type="installed_client",
        benchmark_kit_id="swebench",
        status="completed",
        agent_output={
            "agent_trace": [
                {
                    "trace_step": 1,
                    "trace_role": "assistant",
                    "name": "agent_response",
                    "output": {"answer": "I could not produce a valid patch."},
                }
            ]
        },
        artifact_paths={"submission_patch": "artifacts/submission.patch"},
        runtime_state={},
    )

    normalized = _normalize_judge_output(
        {
            "resolved": False,
            "failure_reason": "patch_apply_failed",
            "diagnostic_reason": "patch_apply_failed",
            "diagnostic_details": {"log_dir": "/tmp/swebench-logs"},
            "log_dir": "/tmp/swebench-logs",
        },
        scheduler_result=scheduler_result,
        judge_source="swebench.verifier_adapter.run",
    )

    assert normalized["diagnostic_reason"] == "patch_apply_failed"
    assert normalized["diagnostic_details"]["agent_trace_summary"]["step_count"] == 1
    assert normalized["diagnostic_details"]["log_dir"] == "/tmp/swebench-logs"
