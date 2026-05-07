from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gage_eval.agent_eval_kits.tau2.artifacts import persist_tau2_artifacts
from gage_eval.agent_eval_kits.tau2.judge.bridges import build_tau2_verifier_request
from gage_eval.config.agentkit_v2 import (
    load_agentkit_v2_config_payload,
    materialize_agentkit_v2_runtime_config_payload,
    resolve_agentkit_v2_runtime_binding_specs,
)
from gage_eval.environment.providers.registry import create_default_provider_registry

from ._support import REPO_ROOT, read_yaml


class _Tau2Runtime:
    def get_state(self) -> dict:
        return {
            "messages": [{"role": "agent", "content": "done"}],
            "agent_cost": 0.25,
            "user_cost": 0.5,
            "reward": 1.0,
            "termination_reason": "agent_stop",
        }


class _Tau2Lease:
    environment = _Tau2Runtime()


@pytest.mark.io
def test_tau2_local_process_config_and_artifact_diagnostics(tmp_path: Path) -> None:
    config_path = REPO_ROOT / "config/custom/tau2/v2_tau2_telecom_local_process.yaml"

    materialized = load_agentkit_v2_config_payload(config_path)
    raw_payload = read_yaml(config_path)
    runtime_config = materialize_agentkit_v2_runtime_config_payload(
        raw_payload,
        config_path,
    )
    binding = resolve_agentkit_v2_runtime_binding_specs(
        materialized,
        runtime_config=runtime_config,
    )["tau2_dut"]

    assert binding.environment_provider == "local_process"
    assert binding.environment_profile["profile_id"] == "tau2-local-process"
    assert binding.provider_config["respond_tool_name"] == "respond"
    registry = create_default_provider_registry()
    assert "tau2" not in registry.registered_provider_ids()
    assert registry.get("local_process").__class__.__name__ == "LocalProcessEnvironmentProvider"
    assert runtime_config["benchmarks"][0]["config"]["domain"] == "telecom"
    assert runtime_config["dut_agents"][0]["trial_policy"] == {"trials": 1}

    artifacts_dir = tmp_path / "sample" / "artifacts"
    session = SimpleNamespace(artifact_layout={"artifacts_dir": str(artifacts_dir)})
    paths = persist_tau2_artifacts(
        session=session,
        scheduler_output={
            "agent_trace": [{"event": "respond"}],
            "runtime_state": {"reward": 1.0},
        },
        environment_lease=_Tau2Lease(),
    )

    assert paths["tau2_state"] == "artifacts/tau2_state.json"
    assert paths["tau2_trajectory"] == "artifacts/tau2_trajectory.json"
    assert paths["tau2_cost"] == "artifacts/tau2_cost.json"
    state = json.loads((artifacts_dir / "tau2_state.json").read_text(encoding="utf-8"))
    trajectory = json.loads((artifacts_dir / "tau2_trajectory.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((artifacts_dir / "tau2_cost.json").read_text(encoding="utf-8"))
    assert state["termination_reason"] == "agent_stop"
    assert state["reward"] == 1.0
    assert trajectory["events"] == [{"role": "agent", "content": "done"}]
    assert diagnostics["agent_cost"] == 0.25
    verifier_request = build_tau2_verifier_request(
        sample_id="tau2-sample-1",
        sample={"id": "tau2-sample-1", "metadata": {"tau2": {"domain": "telecom"}}},
        scheduler_result={
            "artifact_paths": {
                **paths,
                "trace": "artifacts/trace.jsonl",
            },
            "runtime_state": state,
        },
        runtime_context={
            "environment_lease": _Tau2Lease(),
            "trace_events": [
                {"event_type": "model.request", "payload": {"turn_index": 1}},
                {"event_type": "model.response", "payload": {"turn_index": 1}},
                {"event_type": "tool.call.normalized", "payload": {"turn_index": 1}},
                {"event_type": "tool.result", "payload": {"tool_call_id": "call-1"}},
                {"event_type": "tool.result.injected", "payload": {"tool_call_id": "call-1"}},
            ]
        },
    )
    assert verifier_request["runtime_state"]["reward"] == 1.0
    assert verifier_request["trajectory_ref"] == "artifacts/tau2_trajectory.json"
    assert verifier_request["runtime_state_ref"] == "artifacts/tau2_state.json"
    assert verifier_request["trace_ref"] == "artifacts/trace.jsonl"
    assert verifier_request["tool_trace_summary"]["valid"] is True
    assert verifier_request["tool_trace_summary"]["turn_count"] == 1
