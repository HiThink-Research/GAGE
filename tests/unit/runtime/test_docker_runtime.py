import subprocess

import pytest

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
