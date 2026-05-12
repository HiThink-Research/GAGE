from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
from gage_eval.agent_runtime.resources.contracts import ResourceLease
from gage_eval.agent_runtime.schedulers.installed_client import InstalledClientScheduler
from gage_eval.agent_runtime.session import AgentRuntimeSession

from ._support import REPO_ROOT, load_lowered_pipeline_config, role_adapter_by_id


class _CapturingClient:
    def __init__(self) -> None:
        self.environments: list[dict] = []

    def run(self, request: dict, environment: dict) -> dict:
        del request
        self.environments.append(dict(environment))
        return {"answer": "done", "status": "completed"}


@pytest.mark.io
def test_installed_client_docker_config_and_external_handle_only(tmp_path: Path) -> None:
    config_path = REPO_ROOT / "config/custom/swebench_pro/v2_installed_client_docker_smoke.yaml"
    materialized = load_lowered_pipeline_config(config_path)
    role_adapter = role_adapter_by_id(materialized, "swebench_dut")
    params = role_adapter["params"]

    assert role_adapter["agent_runtime_id"] == "swebench_framework_loop"
    assert role_adapter["backend_id"] == "lmstudio_litellm"
    assert params["environment_profile"]["provider"] == "docker"
    assert params["provider_config"]["workdir"] == "/workspace"
    assert params["provider_config"]["exec_workdir"] == "/app"
    assert params["provider_config"]["network_policy"] == "block"

    client = _CapturingClient()
    scheduler = InstalledClientScheduler(client)
    session = AgentRuntimeSession(
        session_id="session-1",
        run_id="run-installed",
        task_id="swebench_pro_smoke",
        sample_id="sample-1",
        benchmark_kit_id="swebench",
        scheduler_type="installed_client",
        client_id="codex",
        artifact_layout={
            "sample_root": str(tmp_path / "sample"),
            "artifacts_dir": str(tmp_path / "sample" / "artifacts"),
            "verifier_result": str(tmp_path / "sample" / "verifier" / "result.json"),
            "runtime_metadata": str(tmp_path / "sample" / "runtime_metadata.json"),
            "raw_error": str(tmp_path / "sample" / "logs" / "raw_error.json"),
        },
        resource_lease=ResourceLease(
            lease_id="lease-1",
            resource_kind="docker",
            profile_id="swebench_runtime",
            lifecycle="per_sample",
            handle_ref={"container_id": "raw-container-id"},
            metadata={
                "environment_profile": {
                    "profile_id": "swebench_runtime",
                    "provider": "docker",
                    "config": {"workdir": "/workspace", "exec_workdir": "/workspace"},
                },
                "provider_config": {"workdir": "/workspace", "exec_workdir": "/workspace"},
            },
        ),
    )

    result = asyncio.run(
        scheduler.arun(
            session=session,
            sample={"id": "sample-1", "instruction": "fix"},
            payload={},
            workflow_bundle=SchedulerWorkflowBundle(
                bundle_id="swebench.installed_client",
                benchmark_kit_id="swebench",
                scheduler_type="installed_client",
                prepare_inputs=lambda **_: {"instruction": "fix"},
                failure_normalizer=lambda **_: {},
            ),
            sandbox_provider=object(),
        )
    )

    assert result.status == "completed"
    environment = client.environments[0]
    assert environment["environment_handle"] == environment["external_environment_handle"]
    assert environment["environment_handle"]["provider"] == "docker"
    assert environment["environment_handle"]["transport"] == "mounted_workdir"
    assert environment["environment_handle"]["workdir"] == "/workspace"
    assert "sandbox_provider" not in environment
    assert "runtime_handle" not in environment
    assert "raw-container-id" not in json.dumps(environment)
