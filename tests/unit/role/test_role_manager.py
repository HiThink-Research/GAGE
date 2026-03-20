from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_instance import Role
from gage_eval.role.runtime.invocation import RoleSessionStore, SampleExecutionContext
from gage_eval.role.role_manager import RoleManager, RoleManagerShutdownError
from gage_eval.role.runtime.sharded_pool import ShardedRolePool


class _StubAdapter(RoleAdapter):
    def __init__(
        self,
        adapter_id: str,
        role_type: str = "dut_model",
        *,
        resource_requirement: dict[str, Any] | None = None,
        fail_on_shutdown: bool = False,
        runtime_mode: str = "native",
        captured_payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            adapter_id=adapter_id,
            role_type=role_type,
            capabilities=(),
            resource_requirement=resource_requirement,
        )
        self.backend = SimpleNamespace(execution_mode=runtime_mode)
        self.shutdown_calls = 0
        self.fail_on_shutdown = fail_on_shutdown
        self.captured_payloads = captured_payloads if captured_payloads is not None else []

    async def ainvoke(
        self, payload: dict[str, Any], state: RoleAdapterState
    ) -> dict[str, Any]:
        del state
        self.captured_payloads.append(dict(payload))
        return {"answer": "ok"}

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.fail_on_shutdown:
            raise RuntimeError(f"adapter failure: {self.adapter_id}")


class _StubPool:
    def __init__(self, pool_id: str, *, fail_on_shutdown: bool = False) -> None:
        self.pool_id = pool_id
        self.fail_on_shutdown = fail_on_shutdown
        self.shutdown_calls = 0

    def acquire(self, timeout: float | None = None):
        del timeout
        return nullcontext(Role(self.pool_id, object()))

    def release(self, role: Role) -> None:
        del role

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.fail_on_shutdown:
            raise RuntimeError(f"pool failure: {self.pool_id}")

    def snapshot(self) -> dict[str, Any]:
        return {
            "pool_type": "stub",
            "adapter_id": self.pool_id,
            "capacity_total": 0,
            "in_use_total": 0,
            "available_total": 0,
            "created_total": 0,
            "healthy": True,
            "shard_count": 0,
            "shards": [],
        }


@pytest.mark.fast
def test_role_manager_uses_composite_pool_for_single_shard_snapshot() -> None:
    manager = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=1)]))
    adapter = _StubAdapter("dut")

    manager.register_role_adapter("dut", adapter)

    assert isinstance(manager._role_pools["dut"], ShardedRolePool)

    initial = manager.snapshot()
    adapter_snapshot = initial["adapters"][0]

    assert initial["snapshot_version"] == "role_manager.v1"
    assert adapter_snapshot["adapter_id"] == "dut"
    assert adapter_snapshot["role_type"] == "dut_model"
    assert adapter_snapshot["runtime_mode"] == "native"
    assert adapter_snapshot["runtime_strategy"] == "native_runtime"
    assert adapter_snapshot["session_mode"] == "explicit_context"
    assert adapter_snapshot["sandbox_enabled_default"] is False
    assert adapter_snapshot["pool_type"] == "sharded"
    assert adapter_snapshot["planned_capacity"] == 1
    assert adapter_snapshot["effective_capacity"] == 1
    assert adapter_snapshot["capacity_total"] == 1
    assert adapter_snapshot["available_total"] == 1
    assert adapter_snapshot["in_use_total"] == 0
    assert adapter_snapshot["created_total"] == 0
    assert adapter_snapshot["shard_count"] == 1

    with manager.borrow_role("dut"):
        active = manager.snapshot()["adapters"][0]
        assert active["in_use_total"] == 1
        assert active["available_total"] == 0
        assert active["created_total"] == 1
        assert active["shards"][0]["in_use"] == 1


@pytest.mark.fast
def test_role_manager_preserves_multi_shard_capacity_without_explicit_pool_size() -> None:
    manager = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=4, cpus=2)]))
    adapter = _StubAdapter(
        "judge",
        role_type="judge_model",
        resource_requirement={
            "gpus": 1,
            "endpoint_list": ["http://judge-a", "http://judge-b"],
            "shard_selection_policy": "round_robin",
        },
    )

    manager.register_role_adapter("judge", adapter)
    snapshot = manager.snapshot()["adapters"][0]

    assert snapshot["adapter_id"] == "judge"
    assert snapshot["shard_count"] == 2
    assert snapshot["planned_capacity"] == 4
    assert snapshot["effective_capacity"] == 4
    assert snapshot["capacity_total"] == 4
    assert snapshot["available_total"] == 4
    assert snapshot["extensions"]["selection_policy"] == "round_robin"
    assert snapshot["extensions"]["fallback_policy"] == "least_in_use"
    assert snapshot["extensions"]["waiting_threads"] == 0
    assert snapshot["extensions"]["policy_fallback_total"] == 0
    assert snapshot["extensions"]["notify_total"] == 0
    assert {shard["metadata"]["endpoint"] for shard in snapshot["shards"]} == {
        "http://judge-a",
        "http://judge-b",
    }


@pytest.mark.fast
def test_role_manager_snapshot_exposes_http_runtime_strategy() -> None:
    manager = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=1)]))
    adapter = _StubAdapter("http_model", runtime_mode="http")

    manager.register_role_adapter("http_model", adapter)

    snapshot = manager.snapshot()["adapters"][0]

    assert snapshot["runtime_mode"] == "http"
    assert snapshot["runtime_strategy"] == "http_runtime"


@pytest.mark.fast
def test_role_manager_legacy_context_bridge_emits_event_and_injects_route_metadata() -> None:
    captured_payloads: list[dict[str, Any]] = []
    manager = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=1)]))
    adapter = _StubAdapter("dut", captured_payloads=captured_payloads)
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="legacy-bridge"))
    sample = {"id": "sample-1", "messages": [{"role": "user", "content": "hello"}]}
    sample_context = SampleExecutionContext(
        sample=sample,
        sample_id="sample-1",
        trace=trace,
        session_store=RoleSessionStore(sample),
    )

    manager.register_role_adapter("dut", adapter)
    token = manager._activate_session(sample_context)
    try:
        with manager.borrow_role("dut") as role:
            result = role.invoke({"sample": sample}, trace)
    finally:
        manager._deactivate_session(token)

    assert result["answer"] == "ok"
    assert captured_payloads
    route_payload = captured_payloads[0]["runtime_route"]
    assert route_payload["session_mode"] == "legacy_contextvar"
    assert route_payload["runtime_mode"] == "native"
    assert captured_payloads[0]["history"][0]["content"] == "hello"
    assert any(
        event["event"] == "legacy_context_bridge_used" for event in trace.events
    )


@pytest.mark.fast
def test_role_manager_shutdown_aggregates_adapter_and_pool_failures() -> None:
    manager = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=1)]))
    failing_adapter = _StubAdapter("bad_adapter", fail_on_shutdown=True)
    healthy_adapter = _StubAdapter("good_adapter")
    failing_pool = _StubPool("bad_pool", fail_on_shutdown=True)
    healthy_pool = _StubPool("good_pool")

    manager._adapters = {
        "bad_adapter": failing_adapter,
        "good_adapter": healthy_adapter,
    }
    manager._role_pools = {
        "bad_pool": failing_pool,
        "good_pool": healthy_pool,
    }

    with pytest.raises(RoleManagerShutdownError) as exc_info:
        manager.shutdown()

    error = exc_info.value

    assert healthy_adapter.shutdown_calls == 1
    assert healthy_pool.shutdown_calls == 1
    assert len(error.issues) == 2
    assert {issue.phase for issue in error.issues} == {
        "adapter_shutdown",
        "pool_shutdown",
    }
    assert {issue.component_id for issue in error.issues} == {
        "bad_adapter",
        "bad_pool",
    }
