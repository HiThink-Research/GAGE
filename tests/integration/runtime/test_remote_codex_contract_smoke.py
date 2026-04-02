from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.compat
@pytest.mark.io
def test_remote_codex_contract_smoke_script_runs_end_to_end(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    result = subprocess.run(
        [sys.executable, str(_dev_root() / "sandbox_smoke" / "contract_smoke_codex.py"), "--workspace-root", str(workspace)],
        capture_output=True,
        text=True,
        check=True,
        env=_python_env(),
        timeout=300,
    )

    payload = _last_json_line(result.stdout)

    assert payload["scheduler_status"] == "success"
    assert payload["verifier_status"] == "passed"
    assert payload["hello_exists"] is True
    assert str(payload["hello_content"]).strip() == "world"
    assert payload["surface_names"] == ["terminal", "fs"]


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    extra_paths = [str(_dev_root()), str(_repo_root() / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        extra_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)
    return env


def _last_json_line(stdout: str) -> dict[str, object]:
    for line in reversed([item.strip() for item in stdout.splitlines() if item.strip()]):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise AssertionError(f"no json payload found in stdout:\n{stdout}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _dev_root() -> Path:
    return _repo_root().parent / "dev"
