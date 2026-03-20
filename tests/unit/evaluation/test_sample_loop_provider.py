from __future__ import annotations

from types import SimpleNamespace
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

    def __init__(
        self,
        runtime_configs: Dict[str, Any] | None = None,
        resources: Dict[str, Any] | None = None,
    ):
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}

    def start(self, config: Dict[str, Any]) -> Dict[str, Any]:
        FakeSandbox.start_calls += 1
        return {"mcp_endpoint": "http://mcp"}

    def teardown(self) -> None:
        FakeSandbox.teardown_calls += 1


class ProviderProbeAdapter(RoleAdapter):
    def __init__(
        self,
        adapter_id: str,
        calls: List[object],
        *,
        sandbox_config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            adapter_id=adapter_id,
            role_type="toolchain",
            capabilities=(),
            sandbox_config=sandbox_config or {"runtime": "fake"},
        )
        self._calls = calls

    async def ainvoke(
        self, payload: Dict[str, Any], state: RoleAdapterState
    ) -> Dict[str, Any]:
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
    resource_profile = ResourceProfile(
        nodes=[NodeResource(node_id="local", gpus=0, cpus=1)]
    )
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


@pytest.mark.fast
def test_sample_loop_routes_distinct_sandbox_providers_per_support_adapter() -> None:
    FakeSandbox.start_calls = 0
    FakeSandbox.teardown_calls = 0
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    calls_a: List[object] = []
    calls_b: List[object] = []
    adapter_a = ProviderProbeAdapter(
        "probe_a",
        calls_a,
        sandbox_config={"runtime": "fake", "sandbox_id": "box_a"},
    )
    adapter_b = ProviderProbeAdapter(
        "probe_b",
        calls_b,
        sandbox_config={"runtime": "fake", "sandbox_id": "box_b"},
    )
    resource_profile = ResourceProfile(
        nodes=[NodeResource(node_id="local", gpus=0, cpus=1)]
    )
    role_manager = RoleManager(resource_profile, concurrency_hint=1)
    role_manager.register_role_adapter("probe_a", adapter_a)
    role_manager.register_role_adapter("probe_b", adapter_b)
    planner = TaskPlanner()
    planner.configure_custom_steps(
        [
            {"step": "support", "adapter_id": "probe_a"},
            {"step": "support", "adapter_id": "probe_b"},
        ]
    )
    sample_loop = SampleLoop(
        [{"id": "s1", "messages": [{"role": "user", "content": "hi"}]}],
        concurrency=1,
        sandbox_manager=manager,
    )
    trace = ObservabilityTrace()

    sample_loop.run(planner=planner, role_manager=role_manager, trace=trace)

    assert calls_a and calls_a[0] is not None
    assert calls_b and calls_b[0] is not None
    assert calls_a[0] is not calls_b[0]
    assert calls_a[0].sandbox_config["sandbox_id"] == "box_a"
    assert calls_b[0].sandbox_config["sandbox_id"] == "box_b"
    assert FakeSandbox.start_calls == 2
    assert FakeSandbox.teardown_calls == 2
    assert not any(
        event["event"] == "legacy_context_bridge_used" for event in trace.events
    )


@pytest.mark.fast
def test_sample_loop_applies_step_scoped_sandbox_route_override() -> None:
    FakeSandbox.start_calls = 0
    FakeSandbox.teardown_calls = 0
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    calls: List[object] = []
    adapter = ProviderProbeAdapter(
        "probe",
        calls,
        sandbox_config={"runtime": "fake", "sandbox_id": "default_box"},
    )
    resource_profile = ResourceProfile(
        nodes=[NodeResource(node_id="local", gpus=0, cpus=1)]
    )
    role_manager = RoleManager(resource_profile, concurrency_hint=1)
    role_manager.register_role_adapter("probe", adapter)
    planner = TaskPlanner()
    planner.configure_custom_steps([{"step": "support", "adapter_id": "probe"}])
    sample_loop = SampleLoop(
        [
            {
                "id": "s1",
                "messages": [{"role": "user", "content": "hi"}],
                "sandbox_routes": {
                    "support.probe": {
                        "sandbox_id": "override_box",
                    }
                },
            }
        ],
        concurrency=1,
        sandbox_manager=manager,
    )
    trace = ObservabilityTrace()

    sample_loop.run(planner=planner, role_manager=role_manager, trace=trace)

    assert calls and calls[0] is not None
    assert calls[0].sandbox_config["sandbox_id"] == "override_box"
    assert any(event["event"] == "runtime_route_selected" for event in trace.events)


@pytest.mark.fast
def test_sample_loop_sets_per_arena_scope_key() -> None:
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    resource_profile = ResourceProfile(
        nodes=[NodeResource(node_id="local", gpus=0, cpus=1)]
    )
    role_manager = RoleManager(resource_profile, concurrency_hint=1)
    arena_adapter = RoleAdapter(
        adapter_id="arena_adapter",
        role_type="arena",
        capabilities=(),
        sandbox_config={
            "runtime": "fake",
            "sandbox_id": "arena_box",
            "lifecycle": "per_arena",
        },
    )
    role_manager.register_role_adapter("arena_adapter", arena_adapter)
    sample_loop = SampleLoop(
        [{"id": "sample-1"}], concurrency=1, sandbox_manager=manager
    )
    plan = SimpleNamespace(
        inference_role=None,
        arena_role="arena_adapter",
        judge_role=None,
        support_steps=[],
        metadata={},
    )
    trace = ObservabilityTrace()

    provider = sample_loop._build_sandbox_provider(
        plan,
        {"id": "sample-1"},
        role_manager,
        trace,
        "sample-1",
    )

    assert provider is not None
    handle = provider.get_handle()
    assert handle is not None
    assert handle.pool_key == "arena_box:arena_adapter_sample-1"
    provider.release()
