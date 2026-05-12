from __future__ import annotations

import ast
import json
from typing import Any, Mapping, Sequence


def load_output_json(payload: bytes | bytearray | str) -> dict[str, Any] | None:
    try:
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8", errors="replace")
        parsed = json.loads(str(payload))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def evaluate_resolution(
    output: Mapping[str, Any],
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
) -> tuple[bool, str | None]:
    tests = output.get("tests") or []
    passed = {item.get("name") for item in tests if isinstance(item, Mapping) and item.get("status") == "PASSED"}
    target = set(fail_to_pass) | set(pass_to_pass)
    if target and not target.issubset(passed):
        return False, "assertion_error"
    if not target:
        return False, "missing_targets"
    return True, None


def score_output(
    output: Mapping[str, Any],
    *,
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
) -> dict[str, Any]:
    fail_tests = list(fail_to_pass)
    pass_tests = list(pass_to_pass)
    resolved, failure_reason = evaluate_resolution(output, fail_tests, pass_tests)
    return {
        "resolved": resolved,
        "score": 1.0 if resolved else 0.0,
        "failure_reason": failure_reason,
        "tests": list(output.get("tests") or []),
        "fail_tests": fail_tests,
        "pass_tests": pass_tests,
    }


def parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            pass
        if "\n" in raw:
            return [item.strip() for item in raw.splitlines() if item.strip()]
        if "," in raw:
            return [item.strip() for item in raw.split(",") if item.strip()]
        return [raw]
    return [str(value)]


_load_output_json = load_output_json
_evaluate_resolution = evaluate_resolution
_parse_list = parse_list
