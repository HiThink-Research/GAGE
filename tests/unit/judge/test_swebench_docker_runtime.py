from __future__ import annotations

from pathlib import Path

import gage_eval.role.judge.swebench_docker as swebench_docker
from gage_eval.role.judge.swebench_docker import SwebenchDocker


class _FakeContainer:
    def __init__(self) -> None:
        self.wait_calls: list[int] = []
        self.kill_calls = 0
        self.remove_calls: list[bool] = []

    def wait(self, timeout: int) -> None:
        self.wait_calls.append(timeout)

    def kill(self) -> None:
        self.kill_calls += 1

    def remove(self, force: bool = False) -> None:
        self.remove_calls.append(force)


class _FakeImages:
    @staticmethod
    def get(image_uri: str) -> object:
        return {"image_uri": image_uri}


class _FakeContainers:
    def __init__(self, *, failure: Exception | None) -> None:
        self.failure = failure
        self.calls: list[tuple[str, dict]] = []
        self.container = _FakeContainer()

    def run(self, image_uri: str, **kwargs):
        self.calls.append((image_uri, kwargs))
        if self.failure is not None and len(self.calls) == 1:
            raise self.failure
        return self.container


class _FakeClient:
    def __init__(self, *, failure: Exception | None) -> None:
        self.images = _FakeImages()
        self.containers = _FakeContainers(failure=failure)


def test_swebench_docker_retries_with_loader_on_amd64_entrypoint_error(monkeypatch, tmp_path) -> None:
    client = _FakeClient(
        failure=RuntimeError('exec: "/bin/bash": cannot execute binary file: unknown')
    )
    monkeypatch.setattr(swebench_docker, "_get_docker_client", lambda: client)

    judge = SwebenchDocker(scripts_dir=str(tmp_path))
    result = judge._run_container(
        image_uri="example/image:latest",
        workspace_dir=tmp_path,
        params={"docker_platform": "linux/amd64", "test_timeout_s": 17},
        run_id="run-1",
        instance_id="instance-1",
    )

    assert result == {"status": "ok"}
    assert len(client.containers.calls) == 2
    assert client.containers.calls[0][1]["entrypoint"] == "/bin/bash"
    assert client.containers.calls[1][1]["entrypoint"] == swebench_docker._AMD64_LOADER_ENTRYPOINTS[0]
    assert client.containers.calls[1][1]["command"] == ["/bin/bash", "-c", "bash /workspace/entryscript.sh"]
    assert client.containers.container.wait_calls == [17]
    assert client.containers.container.kill_calls == 1
    assert client.containers.container.remove_calls == [True]


def test_swebench_docker_loader_retry_is_amd64_only(monkeypatch, tmp_path) -> None:
    client = _FakeClient(
        failure=RuntimeError('exec: "/bin/bash": cannot execute binary file: unknown')
    )
    monkeypatch.setattr(swebench_docker, "_get_docker_client", lambda: client)

    judge = SwebenchDocker(scripts_dir=str(tmp_path))
    result = judge._run_container(
        image_uri="example/image:latest",
        workspace_dir=tmp_path,
        params={"docker_platform": "linux/arm64"},
        run_id="run-2",
        instance_id="instance-2",
    )

    assert result == {"status": "error", "failure_reason": "test_execution_error"}
    assert len(client.containers.calls) == 1
    assert client.containers.calls[0][1]["entrypoint"] == "/bin/bash"


def test_swebench_docker_resolves_sample_scoped_verifier_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))

    paths = swebench_docker._resolve_verifier_artifact_paths(
        {
            "execution_context": {
                "run_id": "run-1",
                "task_id": "task/demo",
                "sample_id": "instance-1",
            }
        },
        {"id": "instance-1", "metadata": {"instance_id": "instance-1"}},
        run_id="run-1",
        instance_id="instance-1",
    )

    assert paths["log_dir"] == Path(tmp_path) / "run-1" / "samples" / "task_demo" / "instance-1" / "verifier" / "logs"
    assert paths["workspace_dir"] == Path(tmp_path) / "run-1" / "samples" / "task_demo" / "instance-1" / "verifier" / "workspace"
    assert paths["result_file"] == Path(tmp_path) / "run-1" / "samples" / "task_demo" / "instance-1" / "verifier" / "result.json"
