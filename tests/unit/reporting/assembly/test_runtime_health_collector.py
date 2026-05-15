from __future__ import annotations

import pytest

from gage_eval.reporting.assembly.health_collector import RuntimeHealthCollector


@pytest.mark.fast
def test_runtime_health_collector_matches_report_step_contract() -> None:
    records = [
        {
            "judge_output": {
                "status": "skipped",
                "failure_code": "verifier.skipped_due_to_scheduler_failure",
            },
            "model_output": {
                "runtime_judge_outcome": {
                    "verifier_input": {
                        "scheduler_result": {
                            "status": "failed",
                            "failure_code": "client_execution.tool_retry_budget_exhausted",
                        }
                    }
                }
            },
        }
    ]

    health = RuntimeHealthCollector().collect(records)

    assert health == {
        "sample_count": 1,
        "completed_count": 0,
        "failed_count": 1,
        "aborted_count": 0,
        "verifier_skipped_count": 1,
        "scheduler_failed_count": 1,
    }


@pytest.mark.fast
def test_runtime_health_collector_treats_plain_model_output_as_completed() -> None:
    records = [{"sample": {"id": "sample-1"}, "model_output": {"answer": "C"}, "metrics": {}}]

    health = RuntimeHealthCollector().collect(records)

    assert health == {
        "sample_count": 1,
        "completed_count": 1,
        "failed_count": 0,
        "aborted_count": 0,
        "verifier_skipped_count": 0,
        "scheduler_failed_count": 0,
    }


@pytest.mark.fast
def test_runtime_health_collector_counts_external_harness_aborted_trials() -> None:
    records = [
        {
            "sample": {"id": "task-1"},
            "judge_output": {"status": {"value": "aborted", "source_trial_id": "trial_0001"}},
            "trial_results": [
                {
                    "trial_id": "trial_0001",
                    "status": "aborted",
                    "failure": {"failure_code": "harbor.trial_exception"},
                }
            ],
        }
    ]

    health = RuntimeHealthCollector().collect(records)

    assert health == {
        "sample_count": 1,
        "completed_count": 0,
        "failed_count": 0,
        "aborted_count": 1,
        "verifier_skipped_count": 0,
        "scheduler_failed_count": 0,
    }
