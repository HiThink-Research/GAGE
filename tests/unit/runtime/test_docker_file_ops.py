from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from gage_eval.sandbox.docker_runtime import DockerSandbox


@pytest.mark.fast
def test_docker_read_file_uses_cp(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[1] == "cp":
            dest = Path(args[-1])
            dest.write_bytes(b"hello")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr("gage_eval.sandbox.docker_runtime.subprocess.run", fake_run)
    monkeypatch.setattr("gage_eval.sandbox.docker_runtime._ensure_docker_available", lambda _bin: None)

    sandbox = DockerSandbox(runtime_configs={"docker_bin": "docker"})
    sandbox._container_id = "cid"

    data = sandbox.read_file("/workspace/output.json")

    assert data == b"hello"
    assert any(call[:2] == ["docker", "cp"] for call in calls)


@pytest.mark.fast
def test_docker_write_file_uses_exec_and_cp(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    captured: dict[str, bytes] = {}

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[1] == "cp":
            temp_path = Path(args[2])
            captured["payload"] = temp_path.read_bytes()
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr("gage_eval.sandbox.docker_runtime.subprocess.run", fake_run)
    monkeypatch.setattr("gage_eval.sandbox.docker_runtime._ensure_docker_available", lambda _bin: None)

    sandbox = DockerSandbox(runtime_configs={"docker_bin": "docker"})
    sandbox._container_id = "cid"

    sandbox.write_file("/workspace/patch.diff", "payload")

    assert captured["payload"] == b"payload"
    assert any(call[:2] == ["docker", "exec"] for call in calls)
    assert any(call[:2] == ["docker", "cp"] for call in calls)
