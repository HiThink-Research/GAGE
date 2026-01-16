from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.io
def test_appworld_verify_script_exists() -> None:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "appworld_verify.sh"
    content = script_path.read_text(encoding="utf-8")

    assert "appworld verify tasks" in content
    assert "APPWORLD_APIS_URL" in content
    assert "APPWORLD_MCP_URL" in content
