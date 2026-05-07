from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gage_eval.agent_eval_kits.tau2.kit import load_kit
from gage_eval.agent_eval_kits.tau2.local_runtime import Tau2LocalEnvironment, Tau2Runtime
from gage_eval.agent_runtime.verifier.contracts import VerifierInput
from gage_eval.environment.lease import EnvironmentLease
from tests._support.stubs.tau2_stub import install_tau2_stub


def _sample() -> dict:
    return {
        "id": "tau2-local-smoke",
        "metadata": {"tau2": {"domain": "telecom", "trial": 0, "seed": 1}},
        "raw_assets": {
            "tau2": {
                "task": {
                    "id": "telecom-1",
                    "user_scenario": {"instructions": "Need device support"},
                    "evaluation_criteria": {"reward_basis": ["DB"]},
                }
            }
        },
    }


@pytest.mark.io
def test_tau2_v2_local_process_bootstrap_and_kit_owned_verifier_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    kit = load_kit()
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
    sample = _sample()

    bootstrap = kit.runtime_entry.bootstrap(
        session=SimpleNamespace(runtime_context={"environment_lease": environment_lease}),
        sample=sample,
        payload={"environment_lease": environment_lease},
        sandbox_provider=None,
    )
    runtime.exec_tool("respond", {"message": "### STOP ###"})
    result = kit.build_verifier_adapter().run(
        VerifierInput(
            benchmark_kit_id="tau2",
            scheduler_type="framework_loop",
            sample_id="tau2-local-smoke",
            sample=sample,
            scheduler_result={"runtime_state": runtime.get_state(), "artifact_paths": {}},
            runtime_context={"environment_lease": environment_lease},
            verifier_resources={},
        )
    )

    assert bootstrap["prompt_context"]["domain"] == "telecom"
    assert kit.build_verifier_adapter().__class__.__module__ == "gage_eval.agent_eval_kits.tau2.judge.adapters"
    assert result.payload["tau2"]["reward"] == 1.0
