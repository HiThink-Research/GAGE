import pytest

from gage_eval.sandbox.remote_runtime import RemoteSandbox


@pytest.mark.fast
def test_remote_sandbox_exec_requester():
    calls = {}

    def requester(url, payload, timeout_s, headers):
        calls["url"] = url
        calls["payload"] = payload
        calls["timeout_s"] = timeout_s
        calls["headers"] = headers
        return {"stdout": "ok", "stderr": "", "exit_code": 0, "duration_ms": 5.0}

    sandbox = RemoteSandbox(
        runtime_configs={
            "requester": requester,
            "exec_path": "run_command",
            "timeout_s": 12,
            "headers": {"x-test": "1"},
        }
    )
    sandbox.start({"data_endpoint": "http://remote/api"})
    result = sandbox.exec("echo ok")
    assert calls["url"] == "http://remote/api/run_command"
    assert calls["payload"]["command"] == "echo ok"
    assert calls["timeout_s"] == 12
    assert calls["headers"]["x-test"] == "1"
    assert result.exit_code == 0
    assert result.stdout == "ok"


@pytest.mark.fast
def test_remote_sandbox_exec_url_override():
    def requester(url, payload, timeout_s, headers):
        return {"result": {"stdout": "ok", "stderr": "", "exit_code": 0}}

    sandbox = RemoteSandbox(runtime_configs={"requester": requester})
    sandbox.start({"exec_url": "http://remote/exec"})
    result = sandbox.exec("ls")
    assert result.exit_code == 0
    assert result.stdout == "ok"
