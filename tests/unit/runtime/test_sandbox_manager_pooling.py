from __future__ import annotations

import pytest

from gage_eval.sandbox.manager import SandboxManager


class FakeSandbox:
    def __init__(self, runtime_configs=None, resources=None):
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}

    def start(self, config):
        return {"env_endpoint": "http://fake"}

    def teardown(self):
        return None


@pytest.mark.fast
def test_sandbox_manager_pool_key_reuse() -> None:
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    config = {"runtime": "fake", "lifecycle": "per_sample", "pool_key": "shared"}
    first = manager.acquire(config)
    manager.release(first)
    second = manager.acquire(config)
    assert id(first.sandbox) == id(second.sandbox)


@pytest.mark.fast
def test_sandbox_manager_per_sample_builds_new_runtime() -> None:
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    config = {"runtime": "fake", "lifecycle": "per_sample"}
    first = manager.acquire(config)
    manager.release(first)
    second = manager.acquire(config)
    assert id(first.sandbox) != id(second.sandbox)
