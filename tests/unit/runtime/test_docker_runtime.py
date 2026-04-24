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
def test_docker_exec_decodes_binary_output(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout=b"\x89PNG", stderr=b"\xff")

    monkeypatch.setattr("gage_eval.sandbox.docker_runtime.subprocess.run", fake_run)
    monkeypatch.setattr("gage_eval.sandbox.docker_runtime._ensure_docker_available", lambda _bin: None)

    sandbox = DockerSandbox(runtime_configs={"docker_bin": "docker"})
    sandbox._container_id = "cid"

    result = sandbox.exec("cat /bin/ls")

    assert result.exit_code == 0
    assert calls[0][-3:] == ["/bin/sh", "-lc", "cat /bin/ls"]
    assert isinstance(result.stdout, str)
    assert isinstance(result.stderr, str)
    assert result.stdout


@pytest.mark.fast
def test_docker_exec_can_skip_login_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("gage_eval.sandbox.docker_runtime.subprocess.run", fake_run)
    monkeypatch.setattr("gage_eval.sandbox.docker_runtime._ensure_docker_available", lambda _bin: None)

    sandbox = DockerSandbox(runtime_configs={"docker_bin": "docker"})
    sandbox._container_id = "cid"

    result = sandbox.exec("bash /workspace/entryscript.sh", login_shell=False)

    assert result.exit_code == 0
    assert calls[0][-3:] == ["/bin/sh", "-c", "bash /workspace/entryscript.sh"]


@pytest.mark.fast
def test_docker_start_requires_image_when_start_container_enabled() -> None:
    sandbox = DockerSandbox(runtime_configs={"start_container": True})

    with pytest.raises(RuntimeError, match="docker_image_missing"):
        sandbox.start({"runtime": "docker", "runtime_configs": {"start_container": True}})


@pytest.mark.fast
def test_docker_describe_runtime_state_includes_exit_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect_payload = """
    [
      {
        "Id": "cid-123",
        "Name": "/gage-sandbox-test",
        "RestartCount": 0,
        "LogPath": "/var/lib/docker/containers/cid-123/cid-123-json.log",
        "Config": {"Image": "jefzda/sweap-images:test"},
        "State": {
          "Status": "exited",
          "Running": false,
          "Paused": false,
          "Restarting": false,
          "OOMKilled": true,
          "Dead": false,
          "Pid": 0,
          "ExitCode": 137,
          "Error": "oom-killed",
          "StartedAt": "2026-04-23T15:00:00Z",
          "FinishedAt": "2026-04-23T15:12:00Z"
        }
      }
    ]
    """.strip()
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(args, 0, stdout=inspect_payload, stderr="")
        if args[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(args, 0, stdout=b"fatal: container exited", stderr=b"")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr("gage_eval.sandbox.docker_runtime.subprocess.run", fake_run)
    monkeypatch.setattr("gage_eval.sandbox.docker_runtime._docker_available", lambda _bin: True)

    sandbox = DockerSandbox(runtime_configs={"docker_bin": "docker"})
    sandbox._container_id = "cid-123"
    sandbox._container_name = "gage-sandbox-test"
    sandbox._running = True

    state = sandbox.describe_runtime_state()

    assert calls[0][:2] == ["docker", "inspect"]
    assert calls[1][:2] == ["docker", "logs"]
    assert state["state_exit_code"] == 137
    assert state["state_oom_killed"] is True
    assert state["state_error"] == "oom-killed"
    assert state["container_name"] == "gage-sandbox-test"
    assert state["image"] == "jefzda/sweap-images:test"
    assert state["logs_tail"] == "fatal: container exited"
