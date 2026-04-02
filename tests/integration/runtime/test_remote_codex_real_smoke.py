from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.compat
@pytest.mark.io
@pytest.mark.network
def test_remote_codex_real_smoke_runs_in_docker(tmp_path: Path) -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker is not available")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")

    _ensure_codex_image()
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-e",
            f"OPENAI_API_KEY={os.environ['OPENAI_API_KEY']}",
            "-e",
            "PYTHONPATH=/workspace/repo/src:/workspace/dev",
            "-v",
            f"{_repo_root()}:/workspace/repo",
            "-v",
            f"{_dev_root()}:/workspace/dev",
            "-w",
            "/workspace",
            "gage-codex-sandbox:latest",
            "python3",
            "/workspace/dev/sandbox_smoke/real_codex_smoke.py",
            "--workspace-root",
            "/tmp/gage-real-codex-smoke",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=1800,
    )

    payload = _last_json_line(result.stdout)

    assert payload["scheduler_status"] == "success"
    assert payload["verifier_status"] == "passed"
    assert payload["codex_version"]
    assert payload["surface_names"] == ["terminal", "fs"]
    assert payload["runtime_handle"]["exec_url"]
    assert isinstance(payload["hello_exists"], bool)


def _ensure_codex_image() -> None:
    subprocess.run(
        [
            "docker",
            "build",
            "-t",
            "gage-codex-sandbox:latest",
            "-f",
            str(_repo_root() / "docker" / "agent_eval" / "codex_sandbox" / "Dockerfile"),
            str(_repo_root()),
        ],
        check=True,
        timeout=1800,
    )


def _last_json_line(stdout: str) -> dict[str, object]:
    for line in reversed([item.strip() for item in stdout.splitlines() if item.strip()]):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise AssertionError(f"no json payload found in stdout:\n{stdout}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _dev_root() -> Path:
    return _repo_root().parent / "dev"
