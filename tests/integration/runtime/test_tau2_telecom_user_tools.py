from __future__ import annotations

from pathlib import Path

from gage_eval.role.context.tau2_bootstrap import Tau2BootstrapContext
from gage_eval.sandbox.manager import SandboxManager
from gage_eval.sandbox.provider import SandboxProvider, SandboxScope
from tests.tau2_stub import install_tau2_stub


def test_tau2_telecom_user_tools(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path, force_user_tool_call=True)
    manager = SandboxManager()
    provider = SandboxProvider(
        manager,
        {"runtime": "tau2", "runtime_configs": {"data_dir": str(tmp_path)}},
        SandboxScope(sample_id="tau2-telecom"),
    )
    sample = {
        "id": "telecom-sample",
        "metadata": {"tau2": {"domain": "telecom", "trial": 0, "seed": 1}},
        "raw_assets": {"tau2": {"task": {"id": "telecom-1", "user_scenario": {"instructions": "Need support"}}}},
    }
    Tau2BootstrapContext().provide({"sample": sample, "sandbox_provider": provider})
    runtime = provider.get_handle().sandbox

    runtime.exec_tool("respond", {"message": "hello"})
    state = runtime.get_state()
    tool_messages = [msg for msg in state["messages"] if getattr(msg, "role", None) == "tool"]
    assert tool_messages, "Expected user tool call to produce tool messages"
