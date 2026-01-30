from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.io
def test_swebench_legacy_configs_removed() -> None:
    config_dir = Path(__file__).resolve().parents[3] / "config" / "custom"
    swebench_configs = sorted(path.name for path in config_dir.glob("swebench*.yaml"))
    assert set(swebench_configs) == {
        "swebench_pro_smoke_agent.yaml",
    }

    legacy_asset_dir = Path(__file__).resolve().parents[3] / "src" / "gage_eval" / "assets" / "judge" / "swebench"
    assert not legacy_asset_dir.exists()
