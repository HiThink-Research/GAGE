from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gage_eval.config.pipeline_config import PipelineConfig


@pytest.mark.io
@pytest.mark.parametrize(
    "config_name",
    ["swebench_pro_smoke_agent.yaml"],
)
def test_swebench_smoke_sandbox_profile(config_name: str) -> None:
    config_path = Path(__file__).resolve().parents[3] / "config" / "custom" / config_name
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(payload)

    profiles = {spec.sandbox_id: spec for spec in config.sandbox_profiles}
    assert "swebench_runtime" in profiles
    profile = profiles["swebench_runtime"]
    assert profile.runtime == "docker"
    runtime_configs = profile.runtime_configs
    assert runtime_configs.get("start_container") is True
    assert runtime_configs.get("network_mode") == "none"
    assert runtime_configs.get("exec_workdir") == "/app"
    volumes = runtime_configs.get("volumes") or []
    if volumes:
        assert any("/run_scripts" in str(volume) for volume in volumes)

    context_adapter = next(spec for spec in config.role_adapters if spec.adapter_id == "swebench_context_provider")
    assert context_adapter.sandbox.get("sandbox_id") == "swebench_runtime"
    assert context_adapter.sandbox.get("lifecycle") == "per_sample"
    assert context_adapter.params.get("implementation") == "swebench_repo"

    judge_adapter = next(spec for spec in config.role_adapters if spec.adapter_id == "swebench_docker_judge")
    assert judge_adapter.sandbox.get("sandbox_id") == "swebench_runtime"
    assert judge_adapter.sandbox.get("lifecycle") == "per_sample"
