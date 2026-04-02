import subprocess
from pathlib import Path

import pytest

import gage_eval.sandbox.docker_runtime as docker_runtime
from gage_eval.sandbox.docker_runtime import DockerSandbox, build_docker_run_command, normalize_runtime_configs


def _arg_value(args: list[str], flag: str) -> str:
    index = args.index(flag)
    return args[index + 1]


@pytest.mark.fast
def test_build_docker_run_command_basic() -> None:
    runtime_configs = normalize_runtime_configs(
        {
            "ports": ["8000:8000"],
            "env": {"APPWORLD_ROOT": "/run"},
            "extra_hosts": ["example:host"],
            "labels": {"gage_eval.managed": "true"},
            "network_mode": "bridge_host",
            "workdir": "/run",
            "user": "1000:1000",
            "command": "echo ok",
        }
    )
    args = build_docker_run_command(
        image="appworld-mcp:latest",
        container_name="gage-test",
        runtime_configs=runtime_configs,
        resources={"cpu": 2, "memory": "4g"},
    )
    assert args[:2] == ["docker", "run"]
    assert "-d" in args
    assert "--rm" in args
    assert _arg_value(args, "--name") == "gage-test"
    assert _arg_value(args, "--network") == "bridge"
    assert "gage_eval.managed=true" in args
    assert "8000:8000" in args
    assert "APPWORLD_ROOT=/run" in args
    assert "example:host" in args
    assert "host.docker.internal:host-gateway" in args
    assert _arg_value(args, "--cpus") == "2"
    assert _arg_value(args, "--memory") == "4g"
    assert args[-3] == "appworld-mcp:latest"
    assert args[-2:] == ["echo", "ok"]


@pytest.mark.fast
def test_build_docker_run_command_host_network_skips_ports() -> None:
    runtime_configs = normalize_runtime_configs({"network_mode": "host", "ports": ["8000:8000"]})
    args = build_docker_run_command(
        image="appworld-mcp:latest",
        container_name="gage-test",
        runtime_configs=runtime_configs,
        resources={},
    )
    assert "-p" not in args


@pytest.mark.fast
def test_build_docker_run_command_resolves_relative_volume_host_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[3])

    args = build_docker_run_command(
        image="gage-codex-sandbox:latest",
        container_name="gage-test",
        runtime_configs={"volumes": [[".", "/workspace"]]},
        resources={},
    )

    volume_index = args.index("-v")
    volume_value = args[volume_index + 1]

    assert volume_value.endswith(":/workspace")
    assert volume_value.startswith(str(Path.cwd()))


@pytest.mark.fast
def test_docker_exec_decodes_binary_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=b"\x89PNG", stderr=b"\xff")

    monkeypatch.setattr("gage_eval.sandbox.docker_runtime.subprocess.run", fake_run)
    monkeypatch.setattr("gage_eval.sandbox.docker_runtime._ensure_docker_available", lambda _bin: None)

    sandbox = DockerSandbox(runtime_configs={"docker_bin": "docker"})
    sandbox._container_id = "cid"

    result = sandbox.exec("cat /bin/ls")

    assert result.exit_code == 0
    assert isinstance(result.stdout, str)
    assert isinstance(result.stderr, str)
    assert result.stdout


@pytest.mark.fast
def test_docker_runtime_start_retries_with_loader_on_amd64_entrypoint_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if len(calls) == 1:
            return subprocess.CompletedProcess(
                args,
                125,
                stdout="",
                stderr='exec: "/bin/bash": cannot execute binary file: unknown',
            )
        return subprocess.CompletedProcess(args, 0, stdout="sandbox-1\n", stderr="")

    monkeypatch.setattr(docker_runtime, "subprocess", docker_runtime.subprocess)
    monkeypatch.setattr("gage_eval.sandbox.docker_runtime.subprocess.run", fake_run)
    monkeypatch.setattr("gage_eval.sandbox.docker_runtime._ensure_docker_available", lambda _bin: None)

    sandbox = DockerSandbox(
        runtime_configs={
            "docker_bin": "docker",
            "platform": "linux/amd64",
            "entrypoint": "/bin/bash",
            "command": ["-c", "sleep 3600"],
            "wait_for_ready": False,
        }
    )

    handle = sandbox.start({"image": "example:latest"})

    assert handle["container_id"] == "sandbox-1"
    assert len(calls) == 2
    assert _arg_value(calls[0], "--entrypoint") == "/bin/bash"
    assert _arg_value(calls[1], "--entrypoint") == docker_runtime._AMD64_LOADER_ENTRYPOINTS[0]
    assert calls[1][-4:] == ["example:latest", "/bin/bash", "-c", "sleep 3600"]


@pytest.mark.fast
def test_docker_runtime_start_does_not_retry_loader_for_non_amd64_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(
            args,
            125,
            stdout="",
            stderr='exec: "/bin/bash": cannot execute binary file: unknown',
        )

    monkeypatch.setattr("gage_eval.sandbox.docker_runtime.subprocess.run", fake_run)
    monkeypatch.setattr("gage_eval.sandbox.docker_runtime._ensure_docker_available", lambda _bin: None)

    sandbox = DockerSandbox(
        runtime_configs={
            "docker_bin": "docker",
            "platform": "linux/arm64",
            "entrypoint": "/bin/bash",
            "command": ["-c", "sleep 3600"],
            "wait_for_ready": False,
        }
    )

    with pytest.raises(RuntimeError, match="docker_run_failed"):
        sandbox.start({"image": "example:latest"})

    assert len(calls) == 1


@pytest.mark.fast
def test_docker_runtime_is_alive_tolerates_inspect_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def fake_run(args, **kwargs):
        recorded["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr("gage_eval.sandbox.docker_runtime.subprocess.run", fake_run)
    monkeypatch.setattr("gage_eval.sandbox.docker_runtime._docker_available", lambda _bin: True)

    sandbox = DockerSandbox(runtime_configs={"docker_bin": "docker"})
    sandbox._running = True
    sandbox._container_id = "cid"

    assert sandbox.is_alive() is True
    assert recorded["timeout"] == docker_runtime._DEFAULT_INSPECT_TIMEOUT_S


@pytest.mark.fast
def test_docker_runtime_is_alive_returns_false_when_inspect_reports_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="missing")

    monkeypatch.setattr("gage_eval.sandbox.docker_runtime.subprocess.run", fake_run)
    monkeypatch.setattr("gage_eval.sandbox.docker_runtime._docker_available", lambda _bin: True)

    sandbox = DockerSandbox(runtime_configs={"docker_bin": "docker"})
    sandbox._running = True
    sandbox._container_id = "cid"

    assert sandbox.is_alive() is False
