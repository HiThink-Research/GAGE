import pytest

from gage_eval.sandbox.base import ExecResult, serialize_exec_result


@pytest.mark.fast
def test_exec_result_serialization():
    result = ExecResult(exit_code=0, stdout="ok", stderr="", duration_ms=1.5)
    payload = result.to_dict()
    assert payload["exit_code"] == 0
    assert payload["stdout"] == "ok"
    assert payload["duration_ms"] == 1.5
    assert serialize_exec_result(result) == payload
