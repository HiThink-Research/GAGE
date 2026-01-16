from pathlib import Path
import subprocess

import pytest


@pytest.mark.io
def test_export_datasets_help() -> None:
    script_path = Path(__file__).resolve().parents[3] / "docker" / "appworld" / "export_datasets.sh"
    result = subprocess.run(
        [str(script_path), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Export AppWorld dataset subsets" in result.stdout
