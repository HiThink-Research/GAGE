from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gage_eval.config.loader import load_pipeline_config_payload


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.fast
def test_custom_configs_do_not_declare_legacy_v1_top_level_resources() -> None:
    forbidden = {"agent_backends", "sandbox_profiles"}
    offenders: list[str] = []

    for config_path in sorted((REPO_ROOT / "config" / "custom").rglob("*.yaml")):
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            continue
        for key in sorted(forbidden.intersection(payload)):
            offenders.append(f"{config_path.relative_to(REPO_ROOT)} declares {key}")

    assert offenders == []


@pytest.mark.fast
def test_normalized_pipeline_payload_omits_empty_legacy_v1_resources() -> None:
    payload = load_pipeline_config_payload(REPO_ROOT / "tests" / "fixtures" / "static_eval" / "aime24_short.yaml")

    assert "agent_backends" not in payload
    assert "sandbox_profiles" not in payload


@pytest.mark.fast
def test_agentkit_v2_surface_no_longer_names_legacy_kind() -> None:
    legacy_kind = "Agent" + "Eval" + "Config"
    offenders: list[str] = []
    roots = [
        REPO_ROOT / "src",
        REPO_ROOT / "tests",
        REPO_ROOT / "config",
        REPO_ROOT / "scripts",
    ]

    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml", ".json"}:
                continue
            content = path.read_text(encoding="utf-8")
            if legacy_kind in content:
                offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []
