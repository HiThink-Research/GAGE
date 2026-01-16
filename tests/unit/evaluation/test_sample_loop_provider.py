from __future__ import annotations

from typing import Any, Dict, List

import pytest

from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.evaluation.task_planner import TaskPlanner
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager
from gage_eval.sandbox.manager import SandboxManager


class FakeSandbox:
    start_calls = 0
    teardown_calls = 0

    def __init__(self, runtime_configs: Dict[str, Any] | None = None, resources: Dict[str, Any] | None = None):
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}

    def start(self, config: Dict[str, Any]) -> Dict[str, Any]:
        FakeSandbox.start_calls += 1
        return {"mcp_endpoint": "http://mcp"}

    def teardown(self) -> None:
        FakeSandbox.teardown_calls += 1


class ProviderProbeAdapter(RoleAdapter):
    def __init__(self, adapter_id: str, calls: List[object]) -> None:
        super().__init__(
            adapter_id=adapter_id,
            role_type="toolchain",
            capabilities=(),
            sandbox_config={"runtime": "fake"},
        )
        self._calls = calls

    async def ainvoke(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:
        provider = payload.get("sandbox_provider")
        self._calls.append(provider)
        if provider:
            handle = provider.get_handle()
            return {"runtime_handle": handle.runtime_handle if handle else {}}
        return {}


@pytest.mark.fast
def test_sample_loop_injects_sandbox_provider() -> None:
    FakeSandbox.start_calls = 0
    FakeSandbox.teardown_calls = 0
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    calls: List[object] = []
    adapter = ProviderProbeAdapter("probe", calls)
    resource_profile = ResourceProfile(nodes=[NodeResource(node_id="local", gpus=0, cpus=1)])
    role_manager = RoleManager(resource_profile, concurrency_hint=1)
    role_manager.register_role_adapter("probe", adapter)
    planner = TaskPlanner()
    planner.configure_custom_steps([{"step": "support", "adapter_id": "probe"}])
    sample_loop = SampleLoop(
        [{"id": "s1", "messages": [{"role": "user", "content": "hi"}]}],
        concurrency=1,
        sandbox_manager=manager,
    )
    trace = ObservabilityTrace()
    sample_loop.run(planner=planner, role_manager=role_manager, trace=trace)

    assert calls and calls[0] is not None
    assert FakeSandbox.start_calls == 1
    assert FakeSandbox.teardown_calls == 1
