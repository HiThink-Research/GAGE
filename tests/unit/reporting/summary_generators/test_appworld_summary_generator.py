from __future__ import annotations

import pytest

from gage_eval.reporting.summary_generators.appworld import AppWorldSummaryGenerator


@pytest.mark.fast
def test_appworld_generator_outputs_v2_result() -> None:
    context = {
        "samples": [
            {
                "sample": {"metadata": {"appworld": {"subset": "train"}}},
                "judge_output": {"appworld": {"tgc": 0.5, "sgc": 1.0}},
            }
        ]
    }

    result = AppWorldSummaryGenerator().generate(context)

    assert result.legacy_payload["appworld_summary"]["overall"]["total"] == 1
    assert result.summary_sections[0]["generator_id"] == "appworld_summary"
    assert result.summary_sections[0]["section_id"] == "overview"


@pytest.mark.fast
def test_appworld_generator_outputs_failed_trial_attention_case() -> None:
    context = {
        "samples": [
            {
                "sample": {
                    "id": "sample-1",
                    "metadata": {"appworld": {"subset": "train"}},
                },
                "trial": {
                    "trial_id": "trial-1",
                    "status": "failed",
                    "failure": {"failure_code": "appworld.missing_success_signal"},
                },
                "judge_output": {"appworld": {"tgc": 0.0, "sgc": 0.0}},
            },
            {
                "sample": {
                    "id": "sample-2",
                    "metadata": {"appworld": {"subset": "train"}},
                },
                "trial": {
                    "trial_id": "trial-2",
                    "status": "failed",
                    "failure": {"failure_code": "appworld.missing_success_signal"},
                },
                "judge_output": {"appworld": {"tgc": 0.0, "sgc": 0.0}},
            },
        ]
    }

    result = AppWorldSummaryGenerator().generate(context)

    case = result.attention_cases[0]
    assert case["case_id"] == "appworld/sample-1/trial-1"
    assert case["sample_id"] == "sample-1"
    assert case["trial_id"] == "trial-1"
    assert case["reason_codes"] == ["appworld.missing_success_signal"]
    assert "appworld.missing_success_signal" in case["summary"]
    assert case["scoring"]["frequency"] == 1.0
    assert case["scoring"]["impact"] == "high"
    assert case["scoring"]["actionability"] == "medium"


@pytest.mark.fast
def test_appworld_generator_normalizes_legacy_missing_success_reason_code() -> None:
    context = {
        "samples": [
            {
                "sample": {
                    "id": "sample-1",
                    "metadata": {"appworld": {"subset": "train"}},
                },
                "trial": {
                    "trial_id": "trial-1",
                    "status": "failed",
                    "failure": {"failure_code": "missing_appworld_success_signal"},
                },
                "judge_output": {"appworld": {"tgc": 0.0, "sgc": 0.0}},
            },
        ]
    }

    result = AppWorldSummaryGenerator().generate(context)

    assert result.attention_cases[0]["reason_codes"] == ["appworld.missing_success_signal"]


@pytest.mark.fast
def test_appworld_generator_outputs_verifier_failure_attention_case() -> None:
    context = {
        "samples": [
            {
                "sample": {
                    "id": "sample-1",
                    "metadata": {"appworld": {"subset": "train"}},
                },
                "trial": {"trial_id": "trial-1", "status": "completed"},
                "judge_output": {
                    "appworld": {"tgc": 0.0, "sgc": 0.0},
                    "verifier_failure": {
                        "failure_code": "verifier.run_verifier.appworld.verifier_failed"
                    },
                },
            }
        ]
    }

    result = AppWorldSummaryGenerator().generate(context)

    assert result.attention_cases[0]["reason_codes"] == [
        "verifier.run_verifier.appworld.verifier_failed"
    ]


@pytest.mark.fast
def test_appworld_generator_reads_live_nested_agentkit_trial_id() -> None:
    context = {
        "samples": [
            {
                "sample": {
                    "id": "appworld-sample-1",
                    "metadata": {"appworld": {"subset": "test"}},
                },
                "model_output": {
                    "agent_eval": {
                        "trial_results": [
                            {
                                "trial_id": "trial_0001",
                                "status": "failed",
                                "failure": {
                                    "failure_code": "appworld.missing_success_signal"
                                },
                            }
                        ]
                    }
                },
                "judge_output": {"appworld": {"tgc": 0.0, "sgc": 0.0}},
            }
        ]
    }

    result = AppWorldSummaryGenerator().generate(context)

    case = result.attention_cases[0]
    assert case["case_id"] == "appworld/appworld-sample-1/trial_0001"
    assert case["trial_id"] == "trial_0001"
    assert case["reason_codes"] == ["appworld.missing_success_signal"]


@pytest.mark.fast
def test_appworld_generator_deduplicates_flat_and_nested_agentkit_trial() -> None:
    trial = {
        "trial_id": "trial_0001",
        "status": "failed",
        "failure": {"failure_code": "appworld.missing_success_signal"},
    }
    context = {
        "samples": [
            {
                "sample": {
                    "id": "appworld_demo_0002",
                    "metadata": {"appworld": {"subset": "test"}},
                },
                "trial_results": [trial],
                "model_output": {"agent_eval": {"trial_results": [dict(trial)]}},
                "judge_output": {"appworld": {"tgc": 0.0, "sgc": 0.0}},
            }
        ]
    }

    result = AppWorldSummaryGenerator().generate(context)

    assert [case["case_id"] for case in result.attention_cases] == [
        "appworld/appworld_demo_0002/trial_0001"
    ]


@pytest.mark.fast
def test_appworld_generator_outputs_attention_for_scheduler_failure_status() -> None:
    context = {
        "samples": [
            {
                "sample": {
                    "id": "appworld-sample-1",
                    "metadata": {"appworld": {"subset": "test"}},
                },
                "trial": {"trial_id": "trial_0001", "status": "completed"},
                "scheduler_result": {
                    "status": "failed",
                    "failure": {"failure_code": "client_execution.tool_retry_budget_exhausted"},
                },
                "judge_output": {
                    "status": "completed",
                    "appworld": {"tgc": 0.0, "sgc": 0.0},
                },
            }
        ]
    }

    result = AppWorldSummaryGenerator().generate(context)

    case = result.attention_cases[0]
    assert case["case_id"] == "appworld/appworld-sample-1/trial_0001"
    assert case["trial_id"] == "trial_0001"
    assert case["reason_codes"] == ["client_execution.tool_retry_budget_exhausted"]


@pytest.mark.fast
def test_appworld_generator_tolerates_judge_output_appworld_without_metadata() -> None:
    context = {
        "samples": [
            {
                "sample": {"id": "appworld-sample-1"},
                "trial": {
                    "trial_id": "trial_0001",
                    "status": "failed",
                    "failure": {"failure_code": "appworld.missing_success_signal"},
                },
                "judge_output": {"appworld": {"tgc": 0.0, "sgc": 0.0}},
            }
        ]
    }

    result = AppWorldSummaryGenerator().generate(context)

    assert result is not None
    assert result.legacy_payload["appworld_summary"]["overall"]["total"] == 1
    assert result.legacy_payload["appworld_summary"]["by_subset"]["unknown"]["total"] == 1
    assert result.attention_cases[0]["case_id"] == "appworld/appworld-sample-1/trial_0001"
