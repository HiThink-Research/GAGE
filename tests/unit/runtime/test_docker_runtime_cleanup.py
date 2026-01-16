from __future__ import annotations

import subprocess

import pytest

from gage_eval.sandbox.docker_runtime import DockerSandbox


@pytest.mark.fast
def test_docker_runtime_per_sample_forces_remove(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr("gage_eval.sandbox.docker_runtime.subprocess.run", fake_run)
    monkeypatch.setattr("gage_eval.sandbox.docker_runtime._docker_available", lambda _bin: True)

    sandbox = DockerSandbox(runtime_configs={"docker_bin": "docker"})
    sandbox._container_id = "cid"
    sandbox._config = {"lifecycle": "per_sample"}

    sandbox.teardown()

    assert any(call[:3] == ["docker", "rm", "-f"] for call in calls)
