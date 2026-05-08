from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gage_eval.agent_eval_kits.tau2.local_runtime import Tau2LocalEnvironment, Tau2Runtime
from gage_eval.agent_eval_kits.tau2.runtime import Tau2RuntimeEntry
from gage_eval.environment.lease import EnvironmentLease
from tests._support.stubs.tau2_stub import install_tau2_stub


def test_tau2_runtime_bootstrap_injects_messages_and_policy(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    runtime = Tau2Runtime(runtime_settings={"data_dir": str(tmp_path)})
    runtime.start({"data_dir": str(tmp_path)})
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
        "id": "bootstrap-sample",
        "metadata": {"tau2": {"domain": "airline", "trial": 0, "seed": 1}},
        "raw_assets": {"tau2": {"task": {"id": "task-bootstrap", "user_scenario": {"instructions": "Help"}}}},
    }
    output = Tau2RuntimeEntry().bootstrap(
        session=SimpleNamespace(runtime_context={"environment_lease": environment_lease}),
        sample=sample,
        payload={"environment_lease": environment_lease},
        sandbox_provider=None,
    )

    assert sample["messages"]
    assert sample["metadata"]["tau2"]["policy"] == "policy"
    assert output["prompt_context"]["tools_schema"]
