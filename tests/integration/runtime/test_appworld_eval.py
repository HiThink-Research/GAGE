from pathlib import Path
import subprocess

import pytest


@pytest.mark.io
def test_appworld_eval_script_help() -> None:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "appworld_eval.sh"
    result = subprocess.run(
        [str(script_path), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "appworld_official_jsonl" in result.stdout
