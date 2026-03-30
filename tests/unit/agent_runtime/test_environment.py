from __future__ import annotations

import base64

import pytest

from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan
from gage_eval.agent_runtime.environment.docker_environment import DockerEnvironment
from gage_eval.agent_runtime.environment.fake import FakeEnvironment
from gage_eval.agent_runtime.environment.provider import EnvironmentProvider
from gage_eval.agent_runtime.environment.remote_environment import RemoteEnvironment
from gage_eval.agent_runtime.resources.remote_sandbox import RemoteSandboxContract
from gage_eval.agent_runtime.spec import AgentRuntimeSpec, ClientSurfacePolicy, ResourcePolicy, SandboxPolicy
from gage_eval.sandbox.base import ExecResult


class DummyDockerSandbox:
    def __init__(self, runtime_configs=None, resources=None):
        self.runtime_configs = dict(runtime_configs or {})
        self.resources = dict(resources or {})
        self.started = False
        self.start_configs: list[dict] = []
        self.commands: list[tuple[str, int]] = []
        self.files: dict[str, bytes] = {}

    def start(self, config):
        self.started = True
        self.start_configs.append(dict(config))
        return {"container_id": "docker-1"}

    def teardown(self):
        self.started = False

    def exec(self, command: str, timeout: int = 30):
        self.commands.append((command, timeout))
        return ExecResult(exit_code=0, stdout=command, stderr="", duration_ms=1.0)

    def read_file(self, path: str) -> bytes:
        return self.files.get(path, b"")

    def write_file(self, path: str, content: bytes) -> None:
        self.files[path] = content


@pytest.mark.fast
def test_fake_environment_exec_and_file_io() -> None:
    env = FakeEnvironment()

    handle = env.start()
    env.write_file("/tmp/output.txt", b"hello")
    result = env.exec("echo ok")
    env.upload_file("/tmp/input.txt", "/workspace/input.txt")

    assert handle == {"fake": True}
    assert env.started is True
    assert result.exit_code == 0
    assert "/workspace/input.txt" in env.files
    assert env.commands == ["echo ok"]


@pytest.mark.fast
def test_docker_environment_delegates_to_sandbox(tmp_path) -> None:
    env = DockerEnvironment(
        config={"runtime_configs": {"labels": {"suite": "test"}}},
        runtime_configs={"docker_bin": "docker"},
        resources={"cpu": 2},
        sandbox_cls=DummyDockerSandbox,
    )

    handle = env.start()
    result = env.exec("pwd", cwd="/workspace", env={"A": "1"}, timeout_sec=7)
    env.write_file("/workspace/file.txt", b"payload")
    data = env.read_file("/workspace/file.txt")
    local = tmp_path / "download.txt"
    env.download_file("/workspace/file.txt", str(local))

    assert handle["container_id"] == "docker-1"
    assert result.stdout.startswith("cd /workspace && A=")
    assert data == b"payload"
    assert local.read_bytes() == b"payload"


@pytest.mark.fast
def test_remote_environment_attached_exec_and_file_roundtrip() -> None:
    calls: list[tuple[str, dict, int, dict, str | None]] = []

    def requester(url, payload, timeout_s, headers, method="POST"):
        calls.append((url, payload, timeout_s, headers, method))
        if method == "POST" and url.endswith("/run_command"):
            if payload["command"].startswith("cat "):
                return {"exit_code": 0, "stdout": "from exec", "stderr": ""}
            return {"exit_code": 0, "stdout": "ok", "stderr": "", "duration_ms": 4.0}
        if method == "POST" and url.endswith("/read_file"):
            return {"content_b64": base64.b64encode(b"hello").decode("ascii")}
        if method == "POST" and url.endswith("/write_file"):
            return {"ok": True}
        raise AssertionError(f"unexpected request: {method} {url}")

    contract = RemoteSandboxContract(
        mode="attached",
        exec_endpoint="http://remote/api/run_command",
        file_endpoint="http://remote/api",
        attach_target="/workspace",
    )
    env = RemoteEnvironment(
        contract=contract,
        runtime_configs={"requester": requester, "headers": {"x-test": "1"}},
    )

    handle = env.start()
    result = env.exec("echo ok")
    data = env.read_file("/tmp/output.txt")
    env.write_file("/tmp/input.txt", b"payload")
    env.probe()

    assert handle["remote_mode"] == "attached"
    assert handle["attach_target"] == "/workspace"
    assert result.stdout == "ok"
    assert data == b"hello"
    assert calls[0][0].endswith("/run_command")
    assert any(url.endswith("/read_file") for url, *_ in calls)
    assert any(url.endswith("/write_file") for url, *_ in calls)


@pytest.mark.fast
def test_remote_environment_managed_control_plane() -> None:
    calls: list[tuple[str, dict, int, dict, str | None]] = []

    def requester(url, payload, timeout_s, headers, method="POST"):
        calls.append((url, payload, timeout_s, headers, method))
        if method == "POST" and url.endswith("/sandboxes"):
            return {
                "sandbox_id": "sbx-1",
                "status": "starting",
                "data_endpoint": "http://remote/data",
                "exec_url": "http://remote/exec",
                "file_read_url": "http://remote/read_file",
                "file_write_url": "http://remote/write_file",
            }
        if method == "GET" and url.endswith("/sandboxes/sbx-1"):
            return {"sandbox_id": "sbx-1", "status": "ready"}
        if method == "DELETE" and url.endswith("/sandboxes/sbx-1"):
            return {"status": "deleted"}
        if method == "POST" and url.endswith("/exec"):
            return {"exit_code": 0, "stdout": "ok", "stderr": ""}
        if method == "POST" and url.endswith("/read_file"):
            return {"content_b64": base64.b64encode(b"managed").decode("ascii")}
        if method == "POST" and url.endswith("/write_file"):
            return {"ok": True}
        raise AssertionError(f"unexpected request: {method} {url}")

    contract = RemoteSandboxContract(
        mode="managed",
        control_endpoint="http://platform/v1",
    )
    env = RemoteEnvironment(
        contract=contract,
        runtime_configs={"requester": requester, "auth_type": "bearer", "auth_token": "secret"},
    )

    handle = env.start()
    assert handle["sandbox_id"] == "sbx-1"
    assert env.probe() is True
    result = env.exec("echo ok")
    assert result.stdout == "ok"
    env.stop()

    assert calls[0][4] == "POST"
    assert calls[0][0].endswith("/sandboxes")
    assert any(method == "DELETE" for *_url, _payload, _timeout, _headers, method in calls)


@pytest.mark.fast
def test_environment_provider_selects_backend_types() -> None:
    provider = EnvironmentProvider()

    fake_plan = _make_plan("fake")
    docker_plan = _make_plan("docker")
    remote_plan = _make_plan("remote", remote_mode="attached")

    fake_env = provider.build(fake_plan, {})
    docker_env = provider.build(docker_plan, {})
    remote_env = provider.build(
        remote_plan,
        {
            "runtime_configs": {"requester": lambda *args, **kwargs: {"ok": True}},
            "remote_sandbox": {
                "mode": "attached",
                "exec_endpoint": "http://remote/api/run_command",
                "file_endpoint": "http://remote/api",
            },
        },
    )

    assert isinstance(fake_env, FakeEnvironment)
    assert isinstance(docker_env, DockerEnvironment)
    assert isinstance(remote_env, RemoteEnvironment)
    assert remote_env.contract is not None
    assert remote_env.contract.mode == "attached"


def _make_plan(environment_kind: str, remote_mode: str | None = None):
    spec = AgentRuntimeSpec(
        agent_runtime_id=f"{environment_kind}-rt",
        scheduler="installed_client",
        benchmark_kit_id="swebench",
        resource_policy=ResourcePolicy(environment_kind=environment_kind),
        client_surface_policy=ClientSurfacePolicy(),
        sandbox_policy=SandboxPolicy(remote_mode=remote_mode),
    )
    return CompiledRuntimePlan(
        runtime_spec=spec,
        scheduler_type=spec.scheduler,
        benchmark_kit_id=spec.benchmark_kit_id,
        client_id=None,
        role_adapter_id=None,
        environment_kind=spec.resource_policy.environment_kind,
        required_surfaces=(),
        optional_surfaces=(),
        sandbox_profile_id=None,
        remote_mode=spec.sandbox_policy.remote_mode,
        params={},
    )
