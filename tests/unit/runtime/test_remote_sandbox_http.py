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


@pytest.mark.fast
def test_remote_sandbox_start_teardown_and_health_via_control_plane() -> None:
    calls = []

    def requester(url, payload, timeout_s, headers, method="POST"):
        calls.append((method, url, payload, timeout_s, headers))
        if method == "POST" and url.endswith("/sandboxes"):
            return {
                "sandbox_id": "sbx-1",
                "status": "starting",
                "data_endpoint": "http://remote/data",
                "exec_url": "http://remote/exec",
                "env_endpoint": "http://remote/env",
                "apis_endpoint": "http://remote/apis",
                "mcp_endpoint": "http://remote/mcp",
            }
        if method == "GET" and url.endswith("/sandboxes/sbx-1"):
            return {"sandbox_id": "sbx-1", "status": "ready"}
        if method == "DELETE" and url.endswith("/sandboxes/sbx-1"):
            return {"status": "deleted"}
        raise AssertionError(f"Unexpected request: {method} {url}")

    sandbox = RemoteSandbox(
        runtime_configs={
            "requester": requester,
            "control_endpoint": "https://platform.example/v1",
            "auth_type": "bearer",
            "auth_token": "secret-token",
        }
    )

    handle = sandbox.start({"image": "appworld:latest", "resources": {"cpu": 2}})

    assert handle["sandbox_id"] == "sbx-1"
    assert handle["env_endpoint"] == "http://remote/env"
    assert handle["apis_endpoint"] == "http://remote/apis"
    assert handle["mcp_endpoint"] == "http://remote/mcp"
    assert sandbox.is_alive() is True

    sandbox.teardown()

    assert calls[0][0] == "POST"
    assert calls[0][1].endswith("/sandboxes")
    assert calls[0][4]["Authorization"] == "Bearer secret-token"
    assert any(method == "DELETE" for method, *_ in calls)


@pytest.mark.fast
def test_remote_sandbox_is_alive_without_control_endpoint() -> None:
    sandbox = RemoteSandbox(runtime_configs={})
    sandbox.start({})

    assert sandbox.is_alive() is True
