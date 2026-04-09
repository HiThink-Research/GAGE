from __future__ import annotations

from pathlib import Path

import yaml

from gage_eval.config.pipeline_config import PipelineConfig


def _load_config(path: Path) -> PipelineConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PipelineConfig.from_dict(payload)


def test_appworld_runtime_config_parses() -> None:
    config = _load_config(
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "appworld"
        / "appworld_agent_demo_runtime.yaml"
    )

    dut_agent = next(spec for spec in config.role_adapters if spec.adapter_id == "dut_agent_main")
    assert dut_agent.agent_runtime_id == "appworld_framework_loop"
    assert dut_agent.params.get("pre_hooks") is None
    assert dut_agent.params.get("post_hooks") is None


def test_tau2_runtime_config_drops_bootstrap_support() -> None:
    config = _load_config(
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "tau2"
        / "tau2_telecom_runtime.yaml"
    )

    adapter_ids = {spec.adapter_id for spec in config.role_adapters}
    assert "tau2_bootstrap" not in adapter_ids
    dut_agent = next(spec for spec in config.role_adapters if spec.adapter_id == "tau2_agent")
    assert dut_agent.agent_runtime_id == "tau2_framework_loop"


def test_terminal_and_skillsbench_runtime_smokes_parse() -> None:
    base = Path(__file__).resolve().parents[3] / "config" / "custom"
    terminal = _load_config(base / "terminal_bench" / "terminal_bench_smoke_runtime.yaml")
    skills = _load_config(base / "skillsbench" / "skillsbench_smoke_runtime.yaml")

    terminal_agent = next(spec for spec in terminal.role_adapters if spec.adapter_id == "terminal_agent_main")
    skills_agent = next(spec for spec in skills.role_adapters if spec.adapter_id == "skillsbench_agent_main")
    assert terminal_agent.agent_runtime_id == "terminal_bench_framework_loop"
    assert skills_agent.agent_runtime_id == "skillsbench_framework_loop"
