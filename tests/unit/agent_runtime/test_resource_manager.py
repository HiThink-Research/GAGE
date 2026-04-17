from __future__ import annotations

import pytest

from gage_eval.agent_runtime.resources.manager import RuntimeResourceManager
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.sandbox.manager import SandboxManager


class FakeSandbox:
    start_configs: list[dict] = []

    def __init__(self, runtime_configs=None, resources=None) -> None:
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}
        self.stopped = False

    def start(self, config):
        FakeSandbox.start_configs.append(dict(config))
        return {"container_id": "fake-container"}

    def teardown(self) -> None:
        self.stopped = True


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
def test_runtime_resource_manager_merges_sample_sandbox_override() -> None:
    FakeSandbox.start_configs = []
    sandbox_manager = SandboxManager(
        profiles={
            "swebench_runtime": {
                "sandbox_id": "swebench_runtime",
                "runtime": "fake",
                "resources": {"cpu": 2},
                "runtime_configs": {"network_mode": "none"},
            }
        }
    )
    sandbox_manager.register_runtime("fake", FakeSandbox)
    manager = RuntimeResourceManager(sandbox_manager)

    binding = manager.acquire(
        _build_session(),
        resource_plan={
            "resource_kind": "docker",
            "sandbox_config": {
                "sandbox_id": "swebench_runtime",
                "lifecycle": "per_sample",
            },
        },
        sample={
            "id": "sample-1",
            "sandbox": {
                "sandbox_id": "swebench_runtime",
                "runtime": "fake",
                "lifecycle": "per_sample",
                "image": "fake-image:1",
            },
        },
    )

    assert FakeSandbox.start_configs
    assert FakeSandbox.start_configs[-1]["image"] == "fake-image:1"
    assert binding.resource_lease is not None
    assert binding.resource_lease.handle_ref["container_id"] == "fake-container"

    manager.release(binding)
