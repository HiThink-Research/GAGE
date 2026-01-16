import pytest

from gage_eval.sandbox.manager import SandboxManager


class FakeSandbox:
    def __init__(self, runtime_configs=None, resources=None):
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}
        self.started_with = None
        self.stopped = False
        self.stop_calls = 0

    def start(self, config):
        self.started_with = dict(config)
        return {"mcp_url": "http://fake"}

    def exec(self, command, timeout=30):
        return None

    def teardown(self):
        self.stopped = True
        self.stop_calls += 1


@pytest.mark.fast
def test_sandbox_manager_merge_profiles():
    manager = SandboxManager(
        profiles={
            "demo": {
                "sandbox_id": "demo",
                "runtime": "fake",
                "resources": {"cpu": 1},
                "runtime_configs": {"network_mode": "bridge"},
            }
        }
    )
    manager.register_runtime("fake", FakeSandbox)
    merged = manager.resolve_config({"sandbox_id": "demo"}, {"runtime_configs": {"network_mode": "host"}})
    assert merged["runtime"] == "fake"
    assert merged["resources"]["cpu"] == 1
    assert merged["runtime_configs"]["network_mode"] == "host"


@pytest.mark.fast
def test_sandbox_manager_pool_reuse():
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    config = {"runtime": "fake", "lifecycle": "per_run", "pool_max": 1}
    first = manager.acquire(config)
    manager.release(first)
    second = manager.acquire(config)
    assert id(first.sandbox) == id(second.sandbox)


@pytest.mark.fast
def test_sandbox_manager_shutdown_tears_down_active():
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    handle = manager.acquire({"runtime": "fake"})
    manager.shutdown()
    assert handle.sandbox.stopped is True
    assert handle.sandbox.stop_calls == 1


@pytest.mark.fast
def test_sandbox_manager_release_clears_active():
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    handle = manager.acquire({"runtime": "fake"})
    manager.release(handle)
    manager.shutdown()
    assert handle.sandbox.stop_calls == 1
