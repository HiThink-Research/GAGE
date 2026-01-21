from __future__ import annotations

import pytest

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder
from gage_eval.sandbox.manager import SandboxManager
from gage_eval.sandbox.provider import SandboxProvider, SandboxScope


class FakeSandbox:
    def __init__(self, runtime_configs=None, resources=None):
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}

    def start(self, config):
        return {"container_id": "fake-container", "mcp_endpoint": "http://fake"}

    def teardown(self):
        return None


@pytest.mark.fast
def test_sandbox_provider_emits_events() -> None:
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="sandbox-events"))
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    provider = SandboxProvider(
        manager,
        {"runtime": "fake"},
        SandboxScope(run_id="run-1", task_id="task-1", sample_id="sample-1"),
        trace=trace,
    )

    provider.get_handle()
    provider.get_handle()
    provider.release()

    events = [event["event"] for event in trace.events]
    assert "sandbox_provider_cache_miss" in events
    assert "sandbox_provider_cache_hit" in events
    assert "sandbox_acquire_start" in events
    assert "sandbox_runtime_start" in events
    assert "sandbox_runtime_ready" in events
    assert "sandbox_acquire_end" in events
    assert "sandbox_release_start" in events
    assert "sandbox_release_end" in events

    cache_miss = next(event for event in trace.events if event["event"] == "sandbox_provider_cache_miss")
    assert cache_miss["sample_id"] == "sample-1"
    assert cache_miss["payload"]["sandbox_id"] == "fake"
