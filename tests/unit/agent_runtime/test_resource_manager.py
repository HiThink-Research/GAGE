from __future__ import annotations

from typing import Any

import pytest

from gage_eval.agent_runtime.resources.manager import RuntimeResourceManager
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.environment.lease import EnvironmentLease


class FakeEnvironment:
    env_id = "env-1"
    name = "fake-e2b"
    provider = "e2b"
    capabilities = {}
    metadata = {}

    def __init__(self) -> None:
        self.stopped = False
        self.delete_on_stop: bool | None = None

    async def stop(self, *, delete: bool = True) -> None:
        self.stopped = True
        self.delete_on_stop = delete


class FakeEnvironmentManager:
    def __init__(self) -> None:
        self.acquire_calls: list[dict[str, Any]] = []
        self.release_calls: list[str] = []
        self.environment = FakeEnvironment()

    async def acquire(self, **kwargs: Any) -> EnvironmentLease:
        self.acquire_calls.append(dict(kwargs))
        return EnvironmentLease(
            lease_id="env-lease-1",
            environment=self.environment,
            provider=str(kwargs["provider"]),
            profile_id=str(kwargs["profile_id"]),
            lifecycle=kwargs["lifecycle"],
            exclusive=True,
            metadata=dict(kwargs["metadata"]),
        )

    async def release(self, lease: EnvironmentLease) -> None:
        self.release_calls.append(lease.lease_id)
        await lease.environment.stop(delete=True)


class SandboxManagerThatMustNotAcquire:
    def resolve_config(
        self,
        role_config: dict[str, Any],
        sample_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = dict(role_config or {})
        if sample_config:
            resolved.update(sample_config)
        return resolved

    def acquire(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("E2B resource acquisition must not call SandboxManager.acquire")


def _build_session() -> AgentRuntimeSession:
    return AgentRuntimeSession(
        session_id="session-1",
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        benchmark_kit_id="swebench",
        scheduler_type="framework_loop",
    )


@pytest.mark.fast
def test_runtime_resource_manager_rejects_legacy_sandbox_config() -> None:
    environment_manager = FakeEnvironmentManager()
    manager = RuntimeResourceManager(environment_manager=environment_manager)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="config.legacy_key.sandbox_config"):
        manager.acquire(
            _build_session(),
            resource_plan={"resource_kind": "docker", "sandbox_config": {"profile_id": "legacy"}},
        )


@pytest.mark.fast
def test_runtime_resource_manager_rejects_legacy_sample_sandbox() -> None:
    environment_manager = FakeEnvironmentManager()
    manager = RuntimeResourceManager(environment_manager=environment_manager)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="sample.legacy_key.sandbox"):
        manager.acquire(
            _build_session(),
            resource_plan={
                "resource_kind": "docker",
                "environment_profile": {"profile_id": "swebench_runtime", "provider": "docker"},
            },
            sample={"id": "sample-1", "sandbox": {"image": "legacy-image"}},
        )


@pytest.mark.fast
def test_runtime_resource_manager_uses_provider_config_resolver_for_sample_image_uri() -> None:
    environment_manager = FakeEnvironmentManager()
    manager = RuntimeResourceManager(environment_manager=environment_manager)  # type: ignore[arg-type]

    def provider_config_resolver(*, sample: dict[str, Any], base_provider_config: dict[str, Any]) -> dict[str, Any]:
        image_uri = ((sample.get("metadata") or {}).get("environment_overrides") or {}).get("image_uri")
        if image_uri:
            return {**base_provider_config, "image": image_uri}
        return dict(base_provider_config)

    binding = manager.acquire(
        _build_session(),
        resource_plan={
            "resource_kind": "docker",
            "environment_profile": {
                "profile_id": "swebench_runtime",
                "provider": "docker",
                "config": {"workdir": "/workspace", "exec_workdir": "/app", "network_policy": "block"},
            },
            "provider_config": {"workdir": "/workspace", "exec_workdir": "/app", "network_policy": "block"},
            "provider_config_resolver": provider_config_resolver,
        },
        sample={
            "id": "sample-1",
            "metadata": {"environment_overrides": {"image_uri": "jefzda/sweap-images:sample-1"}},
        },
    )

    acquire_call = environment_manager.acquire_calls[-1]
    assert acquire_call["provider"] == "docker"
    assert acquire_call["provider_config"]["image"] == "jefzda/sweap-images:sample-1"
    assert acquire_call["provider_config"]["workdir"] == "/workspace"
    assert acquire_call["provider_config"]["exec_workdir"] == "/app"
    assert acquire_call["provider_config"]["network_policy"] == "block"
    assert binding.resource_lease is not None
    assert binding.environment_lease is not None
    assert binding.environment_lease.metadata["exec_workdir"] == "/app"
    assert binding.environment_lease.metadata["workdir"] == "/workspace"
    assert binding.resource_lease.metadata["environment_profile"]["provider"] == "docker"

    manager.release(binding)


@pytest.mark.fast
def test_runtime_resource_manager_acquires_e2b_through_environment_manager() -> None:
    environment_manager = FakeEnvironmentManager()
    manager = RuntimeResourceManager(
        SandboxManagerThatMustNotAcquire(),  # type: ignore[arg-type]
        environment_manager=environment_manager,  # type: ignore[arg-type]
    )

    binding = manager.acquire(
        _build_session(),
        resource_plan={
            "resource_kind": "e2b",
            "environment_profile": {
                "provider": "e2b",
                "profile_id": "swebench-e2b-wrapper",
                "config": {"template_id": "gage-swebench-pro-wrapper"},
                "resources": {"network_policy": "block"},
                "capabilities": {"supports_upload_download": True},
                "lifecycle": "per_sample",
            },
            "provider_config": {"template_id": "gage-swebench-pro-wrapper"},
            "resources": {"network_policy": "block"},
            "lifecycle": "per_sample",
        },
    )

    assert environment_manager.acquire_calls
    acquire_call = environment_manager.acquire_calls[-1]
    assert acquire_call["provider"] == "e2b"
    assert acquire_call["profile_id"] == "swebench-e2b-wrapper"
    assert acquire_call["profile"].provider == "e2b"
    assert acquire_call["profile"].config["template_id"] == "gage-swebench-pro-wrapper"
    assert acquire_call["provider_config"]["template_id"] == "gage-swebench-pro-wrapper"
    assert acquire_call["resources"].network_policy == "block"
    assert binding.environment_lease is not None
    assert binding.resource_lease is not None
    assert binding.resource_lease.resource_kind == "e2b"
    assert binding.resource_lease.profile_id == "swebench-e2b-wrapper"
    assert binding.resource_lease.handle_ref["provider"] == "e2b"

    manager.release(binding)

    assert environment_manager.release_calls == ["env-lease-1"]
    assert environment_manager.environment.stopped is True
    assert environment_manager.environment.delete_on_stop is True
