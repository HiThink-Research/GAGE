from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import requests

from gage_eval.agent_runtime.environment.remote_environment import RemoteEnvironment
from gage_eval.agent_runtime.resources.remote_sandbox import RemoteSandboxContract


@pytest.mark.compat
@pytest.mark.io
def test_remote_managed_stub_lifecycle(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _run_python(
        _dev_root() / "remote_codex_mock" / "bootstrap_workspace.py",
        "--workspace-root",
        str(workspace),
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(_dev_root() / "remote_codex_mock" / "managed_stub.py"),
            "--workspace-root",
            str(workspace),
            "--port",
            "18889",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_python_env(),
    )
    try:
        _wait_for_health("http://127.0.0.1:18889/health")
        contract = RemoteSandboxContract(
            mode="managed",
            control_endpoint="http://127.0.0.1:18889/v1",
            workspace_root=str(workspace),
        )
        environment = RemoteEnvironment(contract=contract)

        handle = environment.start()
        environment.write_file("managed.txt", b"hello")
        data = environment.read_file("managed.txt")
        environment.renew(ttl_s=120)
        alive = environment.probe()
        environment.stop()

        assert handle["sandbox_id"]
        assert handle["surface_names"] == ("terminal", "fs")
        assert data == b"hello"
        assert alive is True
        assert (workspace / "managed.txt").read_text(encoding="utf-8") == "hello"
    finally:
        process.terminate()
        process.wait(timeout=10)


def _run_python(script: Path, *args: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        check=True,
        env=_python_env(),
        timeout=120,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _wait_for_health(url: str) -> None:
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=1)
            response.raise_for_status()
            return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError(f"server not healthy: {url}")


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    extra_paths = [str(_dev_root()), str(_repo_root() / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        extra_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)
    return env


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _dev_root() -> Path:
    return _repo_root().parent / "dev"
