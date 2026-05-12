from __future__ import annotations

from gage_eval.agent_eval_kits.swebench.judge.failure_categories import resolve_swebench_failure_category


def test_classifies_context_overflow_when_spillover_threshold_met() -> None:
    assert (
        resolve_swebench_failure_category(
            {
                "failure_code": "client_execution.run_scheduler.swebench.agent_loop_failed",
                "loop_exit_reason": "max_turns",
                "artifact_spillover_count": 3,
            }
        )
        == "context_overflow_from_listing"
    )


def test_does_not_classify_context_overflow_when_spillover_under_threshold() -> None:
    assert (
        resolve_swebench_failure_category(
            {
                "failure_code": "client_execution.run_scheduler.swebench.agent_loop_failed",
                "loop_exit_reason": "max_turns",
                "artifact_spillover_count": 2,
            }
        )
        == "unknown"
    )


def test_does_not_classify_successful_run_as_context_overflow() -> None:
    assert (
        resolve_swebench_failure_category(
            {
                "status": "completed",
                "resolved": True,
                "artifact_spillover_count": 5,
            }
        )
        == "unknown"
    )


def test_classifies_context_overflow_from_large_tool_output() -> None:
    assert (
        resolve_swebench_failure_category(
            {
                "failure_code": "client_execution.run_scheduler.swebench.agent_loop_failed",
                "loop_exit_reason": "max_turns",
                "max_tool_output_bytes": 100_000,
            }
        )
        == "context_overflow_from_listing"
    )


def test_context_overflow_takes_priority_over_endless_file_reading() -> None:
    assert (
        resolve_swebench_failure_category(
            {
                "failure_code": "client_execution.run_scheduler.swebench.agent_loop_failed",
                "loop_exit_reason": "max_turns",
                "artifact_spillover_count": 3,
                "repeated_command_count": 8,
            }
        )
        == "context_overflow_from_listing"
    )


def test_classifies_endless_file_reading_when_repeated_command_hits_max_turns() -> None:
    assert (
        resolve_swebench_failure_category(
            {
                "failure_code": "client_execution.run_scheduler.swebench.agent_loop_failed",
                "loop_exit_reason": "max_turns",
                "repeated_command_count": 8,
            }
        )
        == "endless_file_reading"
    )


def test_endless_file_reading_takes_priority_over_required_tool_retry_budget() -> None:
    assert (
        resolve_swebench_failure_category(
            {
                "failure_code": "client_execution.tool_retry_budget_exhausted",
                "failure_reason": "max_turns",
                "loop_exit_reason": "max_turns",
                "repeated_command_count": 21,
            }
        )
        == "endless_file_reading"
    )


def test_does_not_classify_endless_file_reading_for_cost_limit_exit() -> None:
    assert (
        resolve_swebench_failure_category(
            {
                "failure_code": "client_execution.cost_limit_exceeded",
                "loop_exit_reason": "cost_limit",
                "repeated_command_count": 10,
            }
        )
        == "unknown"
    )


def test_classifies_syntax_error_from_parse_error_signals() -> None:
    assert (
        resolve_swebench_failure_category(
            {
                "failure_code": "client_execution.tool_argument_invalid",
                "parse_error_count": 1,
                "recent_errors": ["SyntaxError: invalid syntax"],
            }
        )
        == "syntax_error"
    )


def test_parse_error_without_syntax_signal_remains_parse_error() -> None:
    assert (
        resolve_swebench_failure_category(
            {
                "failure_code": "client_execution.tool_argument_invalid",
                "failure_reason": "parse_error",
                "parse_error_count": 1,
                "recent_errors": ["JSON decode failed"],
            }
        )
        == "parse_error"
    )


def test_classifies_assertion_error_as_wrong_solution() -> None:
    assert (
        resolve_swebench_failure_category(
            {
                "status": "failed",
                "failure_reason": "assertion_error",
            }
        )
        == "wrong_solution"
    )
