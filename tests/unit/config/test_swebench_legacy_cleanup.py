from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.mark.io
def test_swebench_legacy_configs_removed() -> None:
    config_dir = Path(__file__).resolve().parents[3] / "config" / "custom"
    swebench_configs = sorted(path.name for path in config_dir.glob("swebench*.yaml"))
    assert swebench_configs == []
    assert (config_dir / "swebench_pro" / "swebench_pro_smoke_agent.yaml").exists()

    legacy_asset_dir = Path(__file__).resolve().parents[3] / "src" / "gage_eval" / "assets" / "judge" / "swebench"
    assert not legacy_asset_dir.exists()


def test_swebench_tau2_and_appworld_custom_configs_use_agentkit_v2_wrapper_shape() -> None:
    config_root = Path(__file__).resolve().parents[3] / "config" / "custom"
    config_paths = (
        sorted((config_root / "swebench_pro").glob("*.yaml"))
        + sorted((config_root / "tau2").glob("*.yaml"))
        + sorted((config_root / "appworld").glob("*.yaml"))
    )
    offenders: list[str] = []
    for config_path in config_paths:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        relpath = str(config_path.relative_to(config_root.parents[1]))
        for legacy_key in ("sandbox_profiles", "runtime_configs", "agent_backends"):
            if legacy_key in payload:
                offenders.append(f"{relpath} declares {legacy_key}")
        if payload.get("kind") != "PipelineConfig":
            offenders.append(f"{relpath} is not a PipelineConfig wrapper")
        for required_key in ("agents", "benchmarks", "environments", "dut_agents"):
            if required_key not in payload:
                offenders.append(f"{relpath} missing {required_key}")

    assert offenders == []
