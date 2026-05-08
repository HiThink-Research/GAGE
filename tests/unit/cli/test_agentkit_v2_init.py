from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml

from gage_eval.config.agentkit_v2 import materialize_agentkit_v2_config_payload


RUN_PATH = Path(__file__).resolve().parents[3] / "run.py"
SPEC = importlib.util.spec_from_file_location("gage_eval_run_cli_agentkit_v2_init", RUN_PATH)
assert SPEC and SPEC.loader
gage_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gage_run)

PROFILE_DEFAULT_CONSTANT_KEYS = {"image", "privileged", "cpu", "memory", "network", "network_policy"}


def _run_init(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], kit_id: str) -> dict[str, Any]:
    monkeypatch.setattr(sys, "argv", ["run.py", "init", kit_id])

    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = yaml.safe_load(captured.out)
    assert isinstance(payload, dict)
    return payload


def _materialize(payload: dict[str, Any]) -> dict[str, Any]:
    return materialize_agentkit_v2_config_payload(payload, source_path=Path("generated-by-init.yaml"))


@pytest.mark.fast
def test_agentkit_v2_init_swebench_generates_five_section_yaml(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _run_init(monkeypatch, capsys, "swebench")

    assert payload["kind"] == "AgentEvalConfig"
    assert {"backends", "agents", "benchmarks", "environments", "dut_agents"}.issubset(payload)
    assert all(isinstance(payload[section], list) and payload[section] for section in (
        "backends",
        "agents",
        "benchmarks",
        "environments",
        "dut_agents",
    ))

    materialized = _materialize(payload)
    assert materialized["benchmarks"][0]["kit_id"] == "swebench"


@pytest.mark.fast
def test_agentkit_v2_init_tau2_uses_local_process_default_profile(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _run_init(monkeypatch, capsys, "tau2")

    environment = payload["environments"][0]
    assert environment["provider"] == "local_process"
    assert environment["profile_id"] == "tau2-local-process"


@pytest.mark.fast
def test_agentkit_v2_init_includes_secret_placeholders_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "real-secret-value")

    payload = _run_init(monkeypatch, capsys, "tau2")
    rendered = yaml.safe_dump(payload, sort_keys=False)

    assert payload["backends"][0]["config"]["api_key"] == "${ENV.MODEL_API_KEY}"
    assert "${ENV.MODEL_API_KEY}" in rendered
    assert "real-secret-value" not in rendered


@pytest.mark.fast
def test_agentkit_v2_init_omits_profile_defaults_from_yaml(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _run_init(monkeypatch, capsys, "swebench")

    profile = payload["environments"][0]["profile"]
    assert "asset_dir" in profile
    assert PROFILE_DEFAULT_CONSTANT_KEYS.isdisjoint(profile)
