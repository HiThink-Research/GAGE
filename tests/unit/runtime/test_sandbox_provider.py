from __future__ import annotations

import pytest

from gage_eval.sandbox.manager import SandboxManager
from gage_eval.sandbox.provider import SandboxProvider, SandboxScope


class FakeSandbox:
    start_calls = 0
    teardown_calls = 0

    def __init__(self, runtime_configs=None, resources=None):
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}
        self._alive = True

    def start(self, config):
        FakeSandbox.start_calls += 1
        return {"env_endpoint": "http://env"}

    def teardown(self):
        FakeSandbox.teardown_calls += 1
        self._alive = False

    def is_alive(self, timeout_s: float | None = None) -> bool:  # noqa: ARG002
        return self._alive


@pytest.mark.fast
def test_sandbox_provider_lazy_start_and_release() -> None:
    FakeSandbox.start_calls = 0
    FakeSandbox.teardown_calls = 0
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    provider = SandboxProvider(
        manager,
        {"runtime": "fake"},
        SandboxScope(run_id="run", task_id="task", sample_id="sample"),
    )

    handle = provider.get_handle()
    assert handle is not None
    assert handle.runtime_handle["env_endpoint"] == "http://env"
    assert FakeSandbox.start_calls == 1

    provider.get_handle()
    assert FakeSandbox.start_calls == 1

    provider.release()
    assert FakeSandbox.teardown_calls == 1


@pytest.mark.fast
def test_sandbox_provider_no_config_returns_none() -> None:
    manager = SandboxManager()
    provider = SandboxProvider(manager, None, SandboxScope(run_id="run"))
    assert provider.get_handle() is None


@pytest.mark.fast
def test_sandbox_provider_rebuilds_dead_cached_handle() -> None:
    FakeSandbox.start_calls = 0
    FakeSandbox.teardown_calls = 0
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    provider = SandboxProvider(
        manager,
        {"runtime": "fake"},
        SandboxScope(run_id="run", task_id="task", sample_id="sample"),
    )

    first = provider.get_handle()
    assert first is not None
    assert FakeSandbox.start_calls == 1

    first.sandbox._alive = False  # type: ignore[attr-defined]
    second = provider.get_handle()

    assert second is not None
    assert second is not first
    assert FakeSandbox.start_calls == 2
    assert FakeSandbox.teardown_calls == 1


@pytest.mark.fast
def test_sandbox_provider_builds_per_arena_pool_key() -> None:
    manager = SandboxManager()
    provider = SandboxProvider(
        manager,
        {"runtime": "docker", "sandbox_id": "arena_box", "lifecycle": "per_arena"},
        SandboxScope(
            run_id="run-1",
            task_id="task-1",
            sample_id="sample-1",
            arena_id="arena_adapter_sample-1",
        ),
    )

    config = provider.sandbox_config
    assert config["lifecycle"] == "per_arena"

    handle = provider.get_handle()
    assert handle is not None
    assert handle.pool_key == "arena_box:arena_adapter_sample-1"
    provider.release()
