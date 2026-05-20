from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.io
def test_run_py_exposes_report_pack_flags() -> None:
    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--report-pack" in result.stdout
    assert "--no-report-pack" in result.stdout
