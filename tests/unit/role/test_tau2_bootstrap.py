from __future__ import annotations

from pathlib import Path

from gage_eval.role.context.tau2_bootstrap import Tau2BootstrapContext
from gage_eval.sandbox.manager import SandboxManager
from gage_eval.sandbox.provider import SandboxProvider, SandboxScope
from tests.tau2_stub import install_tau2_stub


def test_tau2_bootstrap_injects_messages_and_policy(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    manager = SandboxManager()
    provider = SandboxProvider(
        manager,
        {"runtime": "tau2", "runtime_configs": {"data_dir": str(tmp_path)}},
        SandboxScope(sample_id="tau2-bootstrap"),
    )
    sample = {
        "id": "bootstrap-sample",
        "metadata": {"tau2": {"domain": "airline", "trial": 0, "seed": 1}},
        "raw_assets": {"tau2": {"task": {"id": "task-bootstrap", "user_scenario": {"instructions": "Help"}}}},
    }
    output = Tau2BootstrapContext().provide({"sample": sample, "sandbox_provider": provider})

    assert sample["messages"]
    assert sample["metadata"]["tau2"]["policy"] == "policy"
    assert "tools_schema" in output
