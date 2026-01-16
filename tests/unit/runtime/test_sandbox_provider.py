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

    def start(self, config):
        FakeSandbox.start_calls += 1
        return {"env_endpoint": "http://env"}

    def teardown(self):
        FakeSandbox.teardown_calls += 1


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
