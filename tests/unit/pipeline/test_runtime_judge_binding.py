from __future__ import annotations

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.judge import JudgeStep


def test_judge_step_prefers_runtime_judge_outcome() -> None:
    step = JudgeStep("legacy_judge")
    trace = ObservabilityTrace()
    sample = {"id": "sample-1"}

    result = step.execute(
        {
            "sample": sample,
            "model_output": {
                "runtime_judge_outcome": {
                    "judge_output": {
                        "status": "completed",
                        "resolved": True,
                        "score": 1.0,
                    }
                }
            },
        },
        role_manager=None,
        trace=trace,
    )

    assert result["status"] == "completed"
    assert result["resolved"] is True
    assert result["score"] == 1.0
    assert sample["eval_result"]["status"] == "completed"
    assert sample["eval_result"]["resolved"] is True


def test_judge_step_intercepts_runtime_failure_and_writes_eval_result() -> None:
    step = JudgeStep("legacy_judge")
    trace = ObservabilityTrace()
    sample = {"id": "sample-2"}

    result = step.execute(
        {
            "sample": sample,
            "model_output": {
                "runtime_judge_outcome": {
                    "failure": {
                        "failure_code": "verifier.run_verifier.appworld.verifier_failed",
                        "failure_domain": "verifier",
                        "summary": "verifier boom",
                        "artifact_paths": {"raw_error": "logs/raw_error.json"},
                    }
                }
            },
        },
        role_manager=None,
        trace=trace,
    )

    assert result["status"] == "failed"
    assert result["resolved"] is False
    assert result["failure_domain"] == "verifier"
    assert sample["eval_result"]["failure_reason"] == "verifier.run_verifier.appworld.verifier_failed"
