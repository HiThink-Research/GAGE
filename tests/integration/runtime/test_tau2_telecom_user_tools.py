from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gage_eval.agent_eval_kits.tau2.local_runtime import Tau2LocalEnvironment, Tau2Runtime
from gage_eval.agent_eval_kits.tau2.runtime import Tau2RuntimeEntry
from gage_eval.environment.lease import EnvironmentLease
from tests._support.stubs.tau2_stub import install_tau2_stub


def test_tau2_telecom_user_tools(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path, force_user_tool_call=True)
    runtime = Tau2Runtime(runtime_configs={"data_dir": str(tmp_path)})
    runtime.start({"runtime_configs": {"data_dir": str(tmp_path)}})
    environment_lease = EnvironmentLease(
        lease_id="tau2-lease",
        environment=Tau2LocalEnvironment(
            env_id="tau2-env",
            name="tau2-env",
            runtime=runtime,
        ),
        provider="local_process",
        profile_id="tau2-local-process",
        lifecycle="per_sample",
        exclusive=True,
    )
    sample = {
        "id": "telecom-sample",
        "metadata": {"tau2": {"domain": "telecom", "trial": 0, "seed": 1}},
        "raw_assets": {"tau2": {"task": {"id": "telecom-1", "user_scenario": {"instructions": "Need support"}}}},
    }
    Tau2RuntimeEntry().bootstrap(
        session=SimpleNamespace(runtime_context={"environment_lease": environment_lease}),
        sample=sample,
        payload={"environment_lease": environment_lease},
        sandbox_provider=None,
    )

    runtime.exec_tool("respond", {"message": "hello"})
    state = runtime.get_state()
    tool_messages = [msg for msg in state["messages"] if getattr(msg, "role", None) == "tool"]
    assert tool_messages, "Expected user tool call to produce tool messages"
