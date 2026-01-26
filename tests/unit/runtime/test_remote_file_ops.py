from __future__ import annotations

import base64

import pytest

from gage_eval.sandbox.remote_runtime import RemoteSandbox


@pytest.mark.fast
def test_remote_file_endpoints() -> None:
    requests: list[tuple[str, dict, int]] = []

    def fake_requester(url, payload, timeout_s, headers):
        requests.append((url, payload, timeout_s))
        if url.endswith("/read_file"):
            return {"content_b64": base64.b64encode(b"hello").decode("ascii")}
        return {"ok": True}

    sandbox = RemoteSandbox(runtime_configs={"data_endpoint": "http://example", "requester": fake_requester})
    sandbox.start({})

    data = sandbox.read_file("/tmp/output.json")
    sandbox.write_file("/tmp/input.txt", b"payload")

    assert data == b"hello"
    assert requests[0][0].endswith("/read_file")
    assert requests[0][1]["path"] == "/tmp/output.json"
    assert requests[1][0].endswith("/write_file")
    assert requests[1][1]["path"] == "/tmp/input.txt"
    assert "content_b64" in requests[1][1]


@pytest.mark.fast
def test_remote_file_exec_fallback() -> None:
    calls: list[str] = []

    def command_runner(command, timeout):
        calls.append(command)
        if command.startswith("cat "):
            return {"exit_code": 0, "stdout": "from exec", "stderr": ""}
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    sandbox = RemoteSandbox(runtime_configs={"command_runner": command_runner})
    sandbox.start({})

    data = sandbox.read_file("/tmp/read.txt")
    sandbox.write_file("/tmp/write.txt", b"payload")

    assert data == b"from exec"
    assert any("base64" in call for call in calls)
