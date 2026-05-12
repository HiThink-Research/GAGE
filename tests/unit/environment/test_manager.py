from __future__ import annotations

import asyncio
from dataclasses import fields
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from gage_eval.environment import (
    DEFAULT_READ_FILE_LIMIT_BYTES,
    EnvironmentCapabilities,
    EnvironmentCreateError,
    EnvironmentExecError,
    EnvironmentPreflightError,
    EnvironmentResources,
    EnvironmentTimeoutError,
    ExecResult,
)
from gage_eval.environment.lease import EnvironmentLease
from gage_eval.environment.manager import EnvironmentManager, EnvironmentManagerError
from gage_eval.environment.profiles import EnvironmentProfile
from gage_eval.environment.providers.registry import ProviderRegistry


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    image: str = "demo:latest"


class FakeEnvironment:
    def __init__(
        self,
        env_id: str,
        *,
        provider: str = "fake",
        capabilities: EnvironmentCapabilities | None = None,
        exec_result: ExecResult | None = None,
    ) -> None:
        self.env_id = env_id
        self.name = env_id
        self.provider = provider
        self.metadata: dict[str, str] = {"env_id": env_id}
        self.capabilities = capabilities or EnvironmentCapabilities(default_user="agent")
        self.exec_result = exec_result or ExecResult(command="", exit_code=0)
        self.stop_calls: list[bool] = []

    async def start(self, *, force_build: bool = False) -> None:
        del force_build

    async def attach(self) -> None:
        pass

    async def stop(self, *, delete: bool = True) -> None:
        self.stop_calls.append(delete)

    async def exec(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_s: int | None = None,
        user: str | None = None,
        shell: Literal["sh", "login", "none"] = "sh",
    ) -> ExecResult:
        del env, cwd, timeout_s, user, shell
        return self.exec_result.model_copy(update={"command": command})

    async def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        del local_path, remote_path

    async def upload_dir(self, local_path: str | Path, remote_path: str) -> None:
        del local_path, remote_path

    async def download_file(self, remote_path: str, local_path: str | Path) -> None:
        del remote_path, local_path

    async def download_dir(self, remote_path: str, local_path: str | Path) -> None:
        del remote_path, local_path

    async def write_file(self, path: str, content: bytes | str) -> None:
        del path, content

    async def read_file(self, path: str, *, max_bytes: int = DEFAULT_READ_FILE_LIMIT_BYTES) -> bytes:
        del path, max_bytes
        return b""

    async def list_files(self, path: str) -> list[Any]:
        del path
        return []

    async def is_file(self, path: str) -> bool:
        del path
        return False

    async def is_dir(self, path: str) -> bool:
        del path
        return False

    async def get_logs(self, *, stream: Literal["stdout", "stderr"] | None = None) -> str:
        del stream
        return ""

    async def describe(self) -> dict[str, Any]:
        return {"env_id": self.env_id, "provider": self.provider, "diagnostic_ref": f"diag:{self.env_id}"}


class FakeProvider:
    def __init__(
        self,
        *,
        environments: list[FakeEnvironment] | None = None,
        create_failures: int = 0,
        preflight_failure: EnvironmentPreflightError | None = None,
        health_results: list[bool] | None = None,
    ) -> None:
        self.environments = environments or [FakeEnvironment("env-1")]
        self.create_failures = create_failures
        self.preflight_failure = preflight_failure
        self.health_results = health_results or [True]
        self.preflight_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.health_calls: list[FakeEnvironment] = []

    async def preflight(self, **request: Any) -> None:
        self.preflight_calls.append(request)
        if self.preflight_failure is not None:
            raise self.preflight_failure

    async def create(self, **request: Any) -> FakeEnvironment:
        self.create_calls.append(request)
        if len(self.create_calls) <= self.create_failures:
            raise EnvironmentCreateError(f"create failed #{len(self.create_calls)}")
        return self.environments[len(self.create_calls) - self.create_failures - 1]

    async def health_check(self, environment: FakeEnvironment) -> bool:
        self.health_calls.append(environment)
        return self.health_results.pop(0) if self.health_results else True


class RecordingTrace:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: str, payload: dict[str, Any], sample_id: str | None = None) -> None:
        self.events.append({"event": event, "payload": payload, "sample_id": sample_id})


class RaisingTrace:
    def emit(self, event: str, payload: dict[str, Any], sample_id: str | None = None) -> None:
        del event, payload, sample_id
        raise RuntimeError("trace sink unavailable")


class RecordingArtifactSink:
    def __init__(self) -> None:
        self.writes: list[dict[str, Any]] = []

    async def write_artifact(
        self,
        owner: str,
        name: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {"owner": owner, "name": name, "content": content, "metadata": metadata or {}}
        self.writes.append(record)
        return {"artifact_id": f"{owner}:{name}", "bytes": len(content.encode("utf-8"))}


def _profile() -> EnvironmentProfile:
    return EnvironmentProfile(profile_id="explicit-profile", provider="fake", config={"kind": "demo"})


def _resources() -> EnvironmentResources:
    return EnvironmentResources(cpu=1.0, memory_gb=2.0, network_policy="block")


def _registry(provider: FakeProvider) -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("fake", provider)
    return registry


def _manager(
    provider: FakeProvider,
    *,
    trace: RecordingTrace | None = None,
    backoff_records: list[dict[str, Any]] | None = None,
    artifact_sink: RecordingArtifactSink | None = None,
) -> EnvironmentManager:
    async def record_backoff(*, provider: str, attempt: int, delay_s: float, error: BaseException) -> None:
        del error
        if backoff_records is not None:
            backoff_records.append({"provider": provider, "attempt": attempt, "delay_s": delay_s})

    return EnvironmentManager(
        registry=_registry(provider),
        trace=trace,
        backoff=record_backoff,
        artifact_sink=artifact_sink,
    )


async def _acquire(
    manager: EnvironmentManager,
    *,
    lifecycle: Literal["per_sample", "per_task"] = "per_sample",
    metadata: dict[str, str] | None = None,
) -> EnvironmentLease:
    return await manager.acquire(
        kit_id="tau2",
        provider="fake",
        profile_id="explicit-profile",
        profile=_profile(),
        provider_config=ProviderConfig(),
        resources=_resources(),
        startup_env={"A": "1"},
        lifecycle=lifecycle,
        metadata=metadata or {"sample_id": "sample-1"},
    )


def test_environment_lease_has_required_public_fields_and_descriptor_diagnostics() -> None:
    lease_fields = {field.name for field in fields(EnvironmentLease)}

    assert {
        "lease_id",
        "environment",
        "provider",
        "profile_id",
        "lifecycle",
        "exclusive",
        "created_at",
    } <= lease_fields
    assert "env_id" not in lease_fields
    assert "capabilities" not in lease_fields


def test_environment_profile_is_strict_provider_neutral_record() -> None:
    profile = _profile()

    assert profile.profile_id == "explicit-profile"
    assert profile.provider == "fake"
    assert profile.config == {"kind": "demo"}

    with pytest.raises(ValidationError):
        EnvironmentProfile(profile_id="x", provider="fake", config={}, unexpected=True)


def test_provider_registry_requires_explicit_registration() -> None:
    provider = FakeProvider()
    registry = ProviderRegistry()
    registry.register("fake", provider)

    assert registry.get("fake") is provider
    with pytest.raises(KeyError):
        registry.get("missing")


def test_manager_per_sample_acquire_creates_exclusive_lease_and_release_stops_environment() -> None:
    provider = FakeProvider(environments=[FakeEnvironment("env-release")])
    manager = _manager(provider)

    lease = asyncio.run(_acquire(manager))
    asyncio.run(manager.release(lease))

    assert lease.provider == "fake"
    assert lease.profile_id == "explicit-profile"
    assert lease.lifecycle == "per_sample"
    assert lease.exclusive is True
    assert lease.environment.stop_calls == [True]
    assert provider.create_calls[0]["profile_id"] == "explicit-profile"
    assert provider.create_calls[0]["lifecycle"] == "per_sample"


def test_manager_per_task_lifecycle_fails_fast() -> None:
    provider = FakeProvider()
    manager = _manager(provider)

    with pytest.raises(EnvironmentManagerError, match="environment.lifecycle.unsupported"):
        asyncio.run(_acquire(manager, lifecycle="per_task"))

    assert provider.preflight_calls == []
    assert provider.create_calls == []


def test_manager_preflight_failure_maps_to_stable_error_code() -> None:
    provider = FakeProvider(preflight_failure=EnvironmentPreflightError("bad profile"))
    manager = _manager(provider)

    with pytest.raises(EnvironmentManagerError, match="environment.preflight_failed"):
        asyncio.run(_acquire(manager))

    assert len(provider.preflight_calls) == 1
    assert provider.create_calls == []


def test_manager_create_failure_without_retry_budget_maps_to_create_failed() -> None:
    class FailingCreateOnceProvider(FakeProvider):
        retry_budget_by_failure = {EnvironmentCreateError: 0}

    provider = FailingCreateOnceProvider(create_failures=1)
    manager = _manager(provider)

    with pytest.raises(EnvironmentManagerError, match="environment.create_failed"):
        asyncio.run(_acquire(manager))

    assert len(provider.create_calls) == 1


def test_manager_retries_create_failed_twice_with_backoff() -> None:
    backoffs: list[dict[str, Any]] = []
    provider = FakeProvider(create_failures=2, environments=[FakeEnvironment("env-after-retry")])
    manager = _manager(provider, backoff_records=backoffs)

    lease = asyncio.run(_acquire(manager))

    assert lease.environment.env_id == "env-after-retry"
    assert len(provider.create_calls) == 3
    assert backoffs == [
        {"provider": "fake", "attempt": 1, "delay_s": 0.1},
        {"provider": "fake", "attempt": 2, "delay_s": 0.2},
    ]


def test_manager_retries_exhausted_maps_to_unavailable_when_no_runtime_fallback() -> None:
    provider = FakeProvider(create_failures=3)
    manager = _manager(provider)

    with pytest.raises(EnvironmentManagerError, match="environment.unavailable"):
        asyncio.run(_acquire(manager))

    assert len(provider.create_calls) == 3


@pytest.mark.parametrize(
    "health_error",
    [
        EnvironmentExecError("health command failed"),
        RuntimeError("sdk health failed"),
    ],
)
def test_manager_stops_created_environment_when_health_check_raises(health_error: Exception) -> None:
    class RaisingHealthProvider(FakeProvider):
        async def health_check(self, environment: FakeEnvironment) -> bool:
            self.health_calls.append(environment)
            raise health_error

    environment = FakeEnvironment("env-health-error")
    provider = RaisingHealthProvider(environments=[environment])
    manager = _manager(provider)

    with pytest.raises(EnvironmentManagerError, match="environment.unavailable") as excinfo:
        asyncio.run(_acquire(manager))

    assert excinfo.value.__cause__ is health_error
    assert environment.stop_calls == [True]


def test_manager_stops_rebuilt_environment_when_rebuild_health_check_raises() -> None:
    class RebuildHealthRaisesProvider(FakeProvider):
        async def health_check(self, environment: FakeEnvironment) -> bool:
            self.health_calls.append(environment)
            if len(self.health_calls) == 1:
                return False
            raise RuntimeError("rebuilt env unhealthy")

    first = FakeEnvironment("env-unhealthy")
    rebuilt = FakeEnvironment("env-rebuilt-health-error")
    provider = RebuildHealthRaisesProvider(environments=[first, rebuilt])
    manager = _manager(provider)

    with pytest.raises(EnvironmentManagerError, match="environment.unavailable") as excinfo:
        asyncio.run(_acquire(manager))

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert first.stop_calls == [True]
    assert rebuilt.stop_calls == [True]


def test_environment_acquire_trace_emit_failure_is_non_fatal_and_does_not_stop_environment() -> None:
    environment = FakeEnvironment("env-trace-failure")
    provider = FakeProvider(environments=[environment])
    manager = _manager(provider, trace=RaisingTrace())  # type: ignore[arg-type]

    lease = asyncio.run(_acquire(manager))

    assert lease.environment is environment
    assert environment.stop_calls == []


def test_manager_create_runtime_error_maps_to_create_failed_without_wrapping_base_exceptions() -> None:
    class RuntimeCreateFailureProvider(FakeProvider):
        async def create(self, **request: Any) -> FakeEnvironment:
            self.create_calls.append(request)
            raise RuntimeError("sdk create failed")

    provider = RuntimeCreateFailureProvider()
    manager = _manager(provider)

    with pytest.raises(EnvironmentManagerError, match="environment.create_failed") as excinfo:
        asyncio.run(_acquire(manager))

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert len(provider.create_calls) == 1


def test_manager_does_not_wrap_keyboard_interrupt_from_provider_create() -> None:
    class KeyboardInterruptProvider(FakeProvider):
        async def create(self, **request: Any) -> FakeEnvironment:
            self.create_calls.append(request)
            raise KeyboardInterrupt()

    provider = KeyboardInterruptProvider()
    manager = _manager(provider)

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(_acquire(manager))

    assert len(provider.create_calls) == 1


def test_manager_create_timeout_maps_to_stable_timeout_code() -> None:
    class TimeoutCreateProvider(FakeProvider):
        async def create(self, **request: Any) -> FakeEnvironment:
            self.create_calls.append(request)
            raise EnvironmentTimeoutError("create timed out")

    provider = TimeoutCreateProvider()
    manager = _manager(provider)

    with pytest.raises(EnvironmentManagerError, match="environment.timeout") as excinfo:
        asyncio.run(_acquire(manager))

    assert isinstance(excinfo.value.__cause__, EnvironmentTimeoutError)
    assert len(provider.create_calls) == 1


def test_manager_health_check_failure_rebuilds_and_writes_trace_event() -> None:
    trace = RecordingTrace()
    first = FakeEnvironment("env-unhealthy")
    rebuilt = FakeEnvironment("env-rebuilt")
    provider = FakeProvider(environments=[first, rebuilt], health_results=[False, True])
    manager = _manager(provider, trace=trace)

    lease = asyncio.run(_acquire(manager, metadata={"sample_id": "sample-1", "actor": "dut-agent"}))

    assert lease.environment is rebuilt
    assert first.stop_calls == [True]
    assert len(provider.create_calls) == 2
    assert [call["profile_id"] for call in provider.create_calls] == ["explicit-profile", "explicit-profile"]
    rebuild_events = [event for event in trace.events if event["event"] == "environment.rebuild"]
    assert len(rebuild_events) == 1
    assert rebuild_events[0]["payload"]["actor"] == "dut-agent"
    assert rebuild_events[0]["payload"]["old_descriptor"]["env_id"] == "env-unhealthy"
    assert rebuild_events[0]["payload"]["new_descriptor"]["env_id"] == "env-rebuilt"


def test_environment_acquire_trace_event_has_actor_and_descriptor() -> None:
    trace = RecordingTrace()
    provider = FakeProvider(environments=[FakeEnvironment("env-trace")])
    manager = _manager(provider, trace=trace)

    lease = asyncio.run(_acquire(manager, metadata={"sample_id": "sample-2", "actor": "scheduler"}))

    assert lease.describe()["environment_descriptor"]["env_id"] == "env-trace"
    acquire_events = [event for event in trace.events if event["event"] == "environment.acquire"]
    assert len(acquire_events) == 1
    payload = acquire_events[0]["payload"]
    assert payload["actor"] == "scheduler"
    assert payload["descriptor"]["provider"] == "fake"
    assert payload["descriptor"]["profile_id"] == "explicit-profile"
    assert payload["descriptor"]["environment_descriptor"]["env_id"] == "env-trace"
    assert payload["descriptor"]["environment_descriptor"]["capabilities"]["default_user"] == "agent"


def test_manager_uses_explicit_profile_id_and_lifecycle_arguments_not_provider_inference() -> None:
    provider = FakeProvider(environments=[FakeEnvironment("env-explicit")])
    manager = _manager(provider)
    profile = EnvironmentProfile(profile_id="provider-suggested", provider="wrong", config={})

    lease = asyncio.run(
        manager.acquire(
            kit_id="tau2",
            provider="fake",
            profile_id="compiled-plan-profile",
            profile=profile,
            provider_config=ProviderConfig(),
            resources=_resources(),
            startup_env={},
            lifecycle="per_sample",
            metadata={"trial_policy": {"environment_scope": "nested-task"}},
        )
    )

    assert lease.profile_id == "compiled-plan-profile"
    assert lease.lifecycle == "per_sample"
    assert provider.create_calls[0]["profile_id"] == "compiled-plan-profile"
    assert provider.create_calls[0]["profile"].profile_id == "provider-suggested"
    assert provider.create_calls[0]["lifecycle"] == "per_sample"
    assert provider.create_calls[0]["metadata"]["trial_policy"]["environment_scope"] == "nested-task"


def test_lease_runtime_proxy_fills_output_artifact_refs_after_truncation() -> None:
    sink = RecordingArtifactSink()
    stdout = "abcdef"
    stderr = "uvwxyz"
    environment = FakeEnvironment(
        "env-artifacts",
        exec_result=ExecResult(command="cmd", exit_code=0, stdout=stdout, stderr=stderr),
    )
    lease = EnvironmentLease(
        lease_id="lease-artifacts",
        environment=environment,
        provider="fake",
        profile_id="explicit-profile",
        lifecycle="per_sample",
        exclusive=True,
        artifact_sink=sink,
        stdout_limit_bytes=4,
        stderr_limit_bytes=3,
    )

    result = asyncio.run(lease.exec("cmd"))

    assert result.stdout == "abcd"
    assert result.stderr == "uvw"
    assert result.truncated is True
    assert result.output_artifact_refs == [
        {"stream": "stdout", "artifact_id": "lease-artifacts:stdout.txt", "bytes": len(stdout)},
        {"stream": "stderr", "artifact_id": "lease-artifacts:stderr.txt", "bytes": len(stderr)},
    ]
    assert [(write["name"], write["content"]) for write in sink.writes] == [
        ("stdout.txt", stdout),
        ("stderr.txt", stderr),
    ]
