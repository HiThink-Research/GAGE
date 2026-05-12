from __future__ import annotations

import json

from gage_eval.agent_eval_kits.swebench.judge.scoring import (
    evaluate_resolution,
    load_output_json,
    parse_list,
    score_output,
)


def test_scoring_parses_output_json_and_preserves_test_lists() -> None:
    output = {
        "tests": [
            {"name": "tests/test_a.py::test_fix", "status": "PASSED"},
            {"name": "tests/test_b.py::test_keep", "status": "PASSED"},
            {"name": "tests/test_c.py::test_other", "status": "FAILED"},
        ]
    }
    parsed = load_output_json(json.dumps(output).encode("utf-8"))

    assert parsed == output
    resolved, failure_reason = evaluate_resolution(
        parsed,
        fail_to_pass=["tests/test_a.py::test_fix"],
        pass_to_pass=["tests/test_b.py::test_keep"],
    )
    assert resolved is True
    assert failure_reason is None

    score = score_output(
        parsed,
        fail_to_pass=["tests/test_a.py::test_fix", "tests/test_missing.py::test_missing"],
        pass_to_pass=["tests/test_b.py::test_keep"],
    )
    assert score["resolved"] is False
    assert score["failure_reason"] == "assertion_error"
    assert score["fail_tests"] == ["tests/test_a.py::test_fix", "tests/test_missing.py::test_missing"]
    assert score["pass_tests"] == ["tests/test_b.py::test_keep"]


def test_parse_list_supports_legacy_shapes() -> None:
    assert parse_list(["a", 1]) == ["a", "1"]
    assert parse_list("['a', 'b']") == ["a", "b"]
    assert parse_list("a,b") == ["a", "b"]
    assert parse_list("a\nb") == ["a", "b"]
    assert parse_list("a") == ["a"]
    assert parse_list(None) == []


def test_empty_target_returns_missing_targets_diverging_from_official() -> None:
    resolved, failure_reason = evaluate_resolution({"tests": []}, fail_to_pass=[], pass_to_pass=[])

    assert resolved is False
    assert failure_reason == "missing_targets"


def test_f2p_passed_but_p2p_failed_is_unresolved() -> None:
    output = {
        "tests": [
            {"name": "tests/test_fix.py::test_fix", "status": "PASSED"},
            {"name": "tests/test_keep.py::test_keep", "status": "FAILED"},
        ]
    }

    resolved, failure_reason = evaluate_resolution(
        output,
        fail_to_pass=["tests/test_fix.py::test_fix"],
        pass_to_pass=["tests/test_keep.py::test_keep"],
    )

    assert resolved is False
    assert failure_reason == "assertion_error"


def test_p2p_passed_but_f2p_failed_is_unresolved() -> None:
    output = {
        "tests": [
            {"name": "tests/test_fix.py::test_fix", "status": "FAILED"},
            {"name": "tests/test_keep.py::test_keep", "status": "PASSED"},
        ]
    }

    resolved, failure_reason = evaluate_resolution(
        output,
        fail_to_pass=["tests/test_fix.py::test_fix"],
        pass_to_pass=["tests/test_keep.py::test_keep"],
    )

    assert resolved is False
    assert failure_reason == "assertion_error"


def test_skipped_and_error_statuses_do_not_count_as_passed() -> None:
    output = {
        "tests": [
            {"name": "tests/test_fix.py::test_fix", "status": "SKIPPED"},
            {"name": "tests/test_keep.py::test_keep", "status": "ERROR"},
        ]
    }

    resolved, failure_reason = evaluate_resolution(
        output,
        fail_to_pass=["tests/test_fix.py::test_fix"],
        pass_to_pass=["tests/test_keep.py::test_keep"],
    )

    assert resolved is False
    assert failure_reason == "assertion_error"


def test_duplicate_test_name_is_counted_as_passed_when_any_entry_passed() -> None:
    output = {
        "tests": [
            {"name": "tests/test_fix.py::test_fix", "status": "FAILED"},
            {"name": "tests/test_fix.py::test_fix", "status": "PASSED"},
        ]
    }

    resolved, failure_reason = evaluate_resolution(
        output,
        fail_to_pass=["tests/test_fix.py::test_fix"],
        pass_to_pass=[],
    )

    assert resolved is True
    assert failure_reason is None
