from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from gage_eval.agent_runtime.spec import AgentRuntimeSpec, ResourcePolicy, SandboxPolicy


@pytest.mark.fast
def test_agent_runtime_spec_defaults() -> None:
    spec = AgentRuntimeSpec(
        agent_runtime_id="runtime-1",
        scheduler="installed_client",
        benchmark_kit_id="swebench",
    )

    assert spec.resource_policy.environment_kind == "docker"
    assert spec.client_surface_policy.required == ()
    assert spec.sandbox_policy.remote_mode is None


@pytest.mark.fast
def test_agent_runtime_spec_frozen() -> None:
    spec = AgentRuntimeSpec(
        agent_runtime_id="runtime-1",
        scheduler="installed_client",
        benchmark_kit_id="swebench",
    )

    with pytest.raises(FrozenInstanceError):
        spec.agent_runtime_id = "other"


@pytest.mark.fast
def test_agent_runtime_spec_requires_id() -> None:
    with pytest.raises(ValueError):
        AgentRuntimeSpec(
            agent_runtime_id="",
            scheduler="installed_client",
            benchmark_kit_id="swebench",
        )


@pytest.mark.fast
def test_resource_policy_defaults() -> None:
    policy = ResourcePolicy()

    assert policy.environment_kind == "docker"
    assert policy.timeout_sec == 1800


@pytest.mark.fast
def test_sandbox_policy_remote_mode() -> None:
    policy = SandboxPolicy(remote_mode="attached")

    assert policy.remote_mode == "attached"
