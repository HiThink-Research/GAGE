from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[3] / "scripts" / "rollback_report_step_split.py"


@pytest.mark.io
def test_rollback_report_step_split_lists_supported_stages() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--list-stages"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "04a-facade" in result.stdout
    assert "04b-collectors" in result.stdout
    assert "04c-context" in result.stdout
    assert "04d-writer" in result.stdout


@pytest.mark.io
def test_rollback_report_step_split_accepts_stage_and_run_dir(tmp_path) -> None:
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--stage",
            "04d-writer",
            "--run-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "stage: 04d-writer" in result.stdout
    assert "summary_exists: True" in result.stdout
    assert "restore direct EvalCache.write_summary" in result.stdout
