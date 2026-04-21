from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from gage_eval.role.judge.tau2_eval import (
    Tau2Evaluate,
    _build_output_payload,
    _build_simulation,
)
from gage_eval.sandbox.manager import SandboxManager
from gage_eval.sandbox.provider import SandboxProvider, SandboxScope
from tests._support.stubs.tau2_stub import install_tau2_stub


def _build_sample() -> dict:
    task = {
        "id": "task-judge",
        "user_scenario": {"instructions": "Stop quickly"},
        "evaluation_criteria": {"reward_basis": ["DB"]},
    }
    return {
        "id": "sample-judge",
        "metadata": {"tau2": {"domain": "airline", "trial": 0, "seed": 1}},
        "raw_assets": {"tau2": {"task": task}},
    }


def test_tau2_judge_integration(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)

    manager = SandboxManager()
    sandbox_config = {"runtime": "tau2", "runtime_configs": {"data_dir": str(tmp_path)}}
    provider = SandboxProvider(manager, sandbox_config, SandboxScope(sample_id="sample-judge"))

    sample = _build_sample()
    handle = provider.get_handle()
    runtime = handle.sandbox
    runtime.initialize_task(sample)
    runtime.exec_tool("respond", {"message": "please stop"})

    judge = Tau2Evaluate()
    output = judge.invoke({"sample": sample, "sandbox_provider": provider})

    assert output["tau2"]["reward"] == 1.0
    assert output["tau2"]["termination_reason"] == "user_stop"


def test_tau2_judge_defaults_missing_termination_reason_to_agent_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)

    manager = SandboxManager()
    sandbox_config = {"runtime": "tau2", "runtime_configs": {"data_dir": str(tmp_path)}}
    provider = SandboxProvider(manager, sandbox_config, SandboxScope(sample_id="sample-judge-missing-term"))

    sample = _build_sample()
    handle = provider.get_handle()
    runtime = handle.sandbox
    runtime.initialize_task(sample)

    judge = Tau2Evaluate()
    output = judge.invoke({"sample": sample, "sandbox_provider": provider})

    assert output["tau2"]["termination_reason"] == "agent_error"
    assert output["tau2"]["reward"] == 0.0


def test_tau2_judge_keeps_unknown_non_empty_termination_reason(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)

    task = SimpleNamespace(id="task-judge")
    simulation = _build_simulation(
        task=task,
        runtime_state={"termination_reason": "new_reason"},
    )

    output = _build_output_payload(
        simulation=simulation,
        reward_info={},
        domain="airline",
        evaluation_type="all",
    )
    assert output["termination_reason"] == "new_reason"
