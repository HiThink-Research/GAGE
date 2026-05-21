from __future__ import annotations

import pytest

from gage_eval.reporting.summary_generators.swebench import SwebenchSummaryGenerator


@pytest.mark.fast
def test_swebench_generator_outputs_unresolved_attention() -> None:
    context = {
        "samples": [
            {
                "sample": {"_dataset_id": "swebench", "metadata": {"repo": "owner/repo"}},
                "judge_output": {"resolved": False, "failure_reason": "tests_failed"},
            }
        ]
    }

    result = SwebenchSummaryGenerator().generate(context)

    assert result.legacy_payload["swebench_summary"]["overall"]["resolve_rate"] == 0.0
    assert result.summary_sections[0]["generator_id"] == "swebench_summary"
    assert result.summary_sections[0]["section_id"] == "overview"
    assert result.attention_cases[0]["reason_codes"] == ["score.low"]
    assert result.attention_cases[0]["summary"] == "SWE-bench sample is unresolved: score.low."


@pytest.mark.fast
def test_swebench_generator_uses_scheduler_failure_reason_for_unresolved_attention() -> None:
    context = {
        "samples": [
            {
                "sample": {"id": "sample-1", "_dataset_id": "swebench", "metadata": {"repo": "owner/repo"}},
                "trial": {"trial_id": "trial-1"},
                "judge_output": {
                    "resolved": False,
                    "failure_code": "verifier.skipped_due_to_scheduler_failure",
                },
                "scheduler_result": {
                    "status": "failed",
                    "failure": {
                        "failure_code": "client_execution.tool_retry_budget_exhausted"
                    },
                },
            }
        ]
    }

    result = SwebenchSummaryGenerator().generate(context)

    case = result.attention_cases[0]
    assert case["case_id"] == "swebench/sample-1"
    assert case["sample_id"] == "sample-1"
    assert case["trial_id"] == "trial-1"
    assert case["reason_codes"] == [
        "client_execution.tool_retry_budget_exhausted",
    ]
    assert "tool retry budget exhausted" in case["summary"]


@pytest.mark.fast
def test_swebench_generator_deduplicates_reason_codes_in_contract_order() -> None:
    context = {
        "samples": [
            {
                "sample": {"id": "sample-1", "_dataset_id": "swebench"},
                "trial": {
                    "trial_id": "trial-1",
                    "failure": {"failure_code": "timeout"},
                    "failure_code": "timeout",
                },
                "verifier_result": {"failure_code": "verifier.skipped_due_to_scheduler_failure"},
                "scheduler_result": {
                    "failure": {"failure_code": "client_execution.tool_retry_budget_exhausted"}
                },
                "model_output": {"failure_code": "runtime.error"},
                "judge_output": {"resolved": False},
            }
        ]
    }

    result = SwebenchSummaryGenerator().generate(context)

    assert result.attention_cases[0]["reason_codes"] == [
        "timeout",
        "verifier.skipped_due_to_scheduler_failure",
        "client_execution.tool_retry_budget_exhausted",
        "runtime.error",
    ]


@pytest.mark.fast
def test_swebench_generator_reads_live_nested_agentkit_trial_failure() -> None:
    context = {
        "samples": [
            {
                "sample": {
                    "id": "django__django-12345",
                    "_dataset_id": "swebench",
                },
                "model_output": {
                    "agent_eval": {
                        "trial_results": [
                            {
                                "trial_id": "trial_0001",
                                "status": "failed",
                                "scheduler_result": {
                                    "failure": {
                                        "failure_code": "client_execution.tool_retry_budget_exhausted"
                                    }
                                },
                            }
                        ]
                    }
                },
                "judge_output": {"resolved": False},
            }
        ]
    }

    result = SwebenchSummaryGenerator().generate(context)

    case = result.attention_cases[0]
    assert case["trial_id"] == "trial_0001"
    assert case["reason_codes"] == ["client_execution.tool_retry_budget_exhausted"]


@pytest.mark.fast
def test_swebench_generator_reads_sample_predict_result_agentkit_trial_failure() -> None:
    context = {
        "samples": [
            {
                "sample": {
                    "id": "django__django-12345",
                    "_dataset_id": "swebench",
                    "predict_result": [
                        {
                            "agent_eval": {
                                "trial_results": [
                                    {
                                        "trial_id": "trial_0001",
                                        "status": "failed",
                                        "failure": {
                                            "failure_code": "client_execution.tool_retry_budget_exhausted"
                                        },
                                    }
                                ]
                            }
                        }
                    ],
                },
                "judge_output": {"resolved": False},
            }
        ]
    }

    result = SwebenchSummaryGenerator().generate(context)

    case = result.attention_cases[0]
    assert case["trial_id"] == "trial_0001"
    assert case["reason_codes"] == ["client_execution.tool_retry_budget_exhausted"]


@pytest.mark.fast
@pytest.mark.parametrize(
    ("model_output", "expected_reason"),
    [
        (
            {
                "runtime_judge_outcome": {
                    "judge_output": {
                        "failure_code": "verifier.skipped_due_to_scheduler_failure"
                    }
                }
            },
            "verifier.skipped_due_to_scheduler_failure",
        ),
        (
            {
                "runtime_judge_outcome": {
                    "verifier_result": {
                        "payload": {
                            "failure_code": "verifier.skipped_due_to_scheduler_failure"
                        }
                    }
                }
            },
            "verifier.skipped_due_to_scheduler_failure",
        ),
        (
            {
                "runtime_judge_outcome": {
                    "verifier_input": {
                        "scheduler_result": {
                            "failure": {
                                "failure_code": "client_execution.tool_retry_budget_exhausted"
                            }
                        }
                    }
                }
            },
            "client_execution.tool_retry_budget_exhausted",
        ),
        (
            {
                "runtime_judge_outcome": {
                    "verifier_input": {
                        "scheduler_result": {
                            "failure_code": "client_execution.tool_retry_budget_exhausted"
                        }
                    }
                }
            },
            "client_execution.tool_retry_budget_exhausted",
        ),
    ],
)
def test_swebench_generator_reads_runtime_judge_outcome_reason_sources(
    model_output: dict[str, object],
    expected_reason: str,
) -> None:
    context = {
        "samples": [
            {
                "sample": {"id": "django__django-12345", "_dataset_id": "swebench"},
                "model_output": model_output,
                "judge_output": {"resolved": False},
            }
        ]
    }

    result = SwebenchSummaryGenerator().generate(context)

    assert result.attention_cases[0]["reason_codes"] == [expected_reason]
    assert result.attention_cases[0]["reason_codes"] != ["score.low"]


@pytest.mark.fast
def test_swebench_generator_reads_sample_predict_result_runtime_judge_reason() -> None:
    context = {
        "samples": [
            {
                "sample": {
                    "id": "django__django-12345",
                    "_dataset_id": "swebench",
                    "predict_result": [
                        {
                            "runtime_judge_outcome": {
                                "verifier_result": {
                                    "payload": {
                                        "failure_code": "verifier.skipped_due_to_scheduler_failure"
                                    }
                                }
                            }
                        }
                    ],
                },
                "judge_output": {"resolved": False},
            }
        ]
    }

    result = SwebenchSummaryGenerator().generate(context)

    assert result.attention_cases[0]["reason_codes"] == [
        "verifier.skipped_due_to_scheduler_failure"
    ]
