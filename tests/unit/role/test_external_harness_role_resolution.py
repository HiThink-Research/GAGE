from __future__ import annotations

from typing import Any, Iterable

import pytest

from gage_eval.config.pipeline_config import RoleAdapterSpec
from gage_eval.config.registry import ConfigRegistry
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState
from gage_eval.role.adapters.harbor import HarborAdapter
from gage_eval.external_harness_kits.base import (
    TaskBatchHarnessHandle,
    TaskBatchHarnessPlan,
    TaskBatchHarnessRequest,
    TaskBatchHarnessResult,
)
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager


class _InvokableAdapter(RoleAdapter):
    def __init__(
        self,
        adapter_id: str,
        *,
        role_type: str = "dut_model",
        capabilities: tuple[str, ...] = (),
    ) -> None:
        super().__init__(adapter_id=adapter_id, role_type=role_type, capabilities=capabilities)
        self.invoke_calls = 0

    async def ainvoke(self, payload: dict[str, Any], state: RoleAdapterState) -> dict[str, Any]:
        del state
        self.invoke_calls += 1
        return {"answer": payload.get("prompt", "ok")}


class _TaskBatchHarnessAdapter(RoleAdapter):
    def __init__(
        self,
        adapter_id: str = "harbor",
        *,
        capabilities: tuple[str, ...] = ("task_batch_harness",),
    ) -> None:
        super().__init__(
            adapter_id=adapter_id,
            role_type="external_harness",
            capabilities=capabilities,
        )

    def translate(self, request: TaskBatchHarnessRequest) -> TaskBatchHarnessPlan:
        return TaskBatchHarnessPlan(adapter_id=self.adapter_id, payload={"request": request})

    def launch(self, plan: TaskBatchHarnessPlan) -> TaskBatchHarnessHandle:
        return TaskBatchHarnessHandle(adapter_id=self.adapter_id, payload={"plan": plan})

    def poll_until_done(self, handle: TaskBatchHarnessHandle) -> TaskBatchHarnessResult:
        return TaskBatchHarnessResult(adapter_id=self.adapter_id, payload={"handle": handle})

    def parse_results(self, result: TaskBatchHarnessResult) -> Iterable[Any]:
        return (result.payload,)

    async def ainvoke(self, payload: dict[str, Any], state: RoleAdapterState) -> dict[str, Any]:
        raise AssertionError("task-batch harness adapter must not enter sample invoke pool")


class _StructuralTaskBatchHarnessAdapter:
    adapter_id = "structural"
    role_type = "external_harness"
    capabilities: tuple[str, ...] = ()
    resource_requirement: dict[str, Any] = {}
    sandbox_config: dict[str, Any] = {}

    def translate(self, request: TaskBatchHarnessRequest) -> TaskBatchHarnessPlan:
        return TaskBatchHarnessPlan(adapter_id=self.adapter_id, payload={"request": request})

    def launch(self, plan: TaskBatchHarnessPlan) -> TaskBatchHarnessHandle:
        return TaskBatchHarnessHandle(adapter_id=self.adapter_id, payload={"plan": plan})

    def poll_until_done(self, handle: TaskBatchHarnessHandle) -> TaskBatchHarnessResult:
        return TaskBatchHarnessResult(adapter_id=self.adapter_id, payload={"handle": handle})

    def parse_results(self, result: TaskBatchHarnessResult) -> Iterable[Any]:
        return (result.payload,)


class _SampleStepExternalWithTaskBatchMethodNames(_InvokableAdapter):
    def __init__(self, adapter_id: str = "external_sample_structural") -> None:
        super().__init__(
            adapter_id,
            role_type="external_harness",
            capabilities=("sample_step_harness",),
        )

    def translate(self, request: TaskBatchHarnessRequest) -> TaskBatchHarnessPlan:
        return TaskBatchHarnessPlan(adapter_id=self.adapter_id, payload={"request": request})

    def launch(self, plan: TaskBatchHarnessPlan) -> TaskBatchHarnessHandle:
        return TaskBatchHarnessHandle(adapter_id=self.adapter_id, payload={"plan": plan})

    def poll_until_done(self, handle: TaskBatchHarnessHandle) -> TaskBatchHarnessResult:
        return TaskBatchHarnessResult(adapter_id=self.adapter_id, payload={"handle": handle})

    def parse_results(self, result: TaskBatchHarnessResult) -> Iterable[Any]:
        return (result.payload,)


def _manager() -> RoleManager:
    return RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=1)]))


@pytest.mark.fast
def test_task_batch_adapter_resolves_through_task_batch_registry() -> None:
    manager = _manager()
    adapter = _TaskBatchHarnessAdapter()

    manager.register_role_adapter("harbor", adapter)

    assert manager.get_task_batch_harness_adapter("harbor") is adapter


@pytest.mark.fast
def test_config_registry_resolves_harbor_adapter_colon_class_path_as_role_adapter() -> None:
    spec = RoleAdapterSpec(
        adapter_id="harbor_tb2",
        role_type="external_harness",
        class_path="gage_eval.role.adapters.harbor:HarborAdapter",
        capabilities=("task_batch_harness",),
        backend_id="lmstudio_qwen",
        env_id="tb2_docker",
        trial_policy={"trials": 1},
        params={"harness": {"agent": {"kind": "base_agent", "name": "terminus-2"}}},
    )

    adapter = ConfigRegistry().resolve_role_adapter(
        spec,
        backends={"lmstudio_qwen": object()},
    )

    assert isinstance(adapter, HarborAdapter)
    assert adapter.backend_id == "lmstudio_qwen"
    assert adapter.env_id == "tb2_docker"
    assert adapter.trial_policy == {"trials": 1}
    assert adapter.params == {"harness": {"agent": {"kind": "base_agent", "name": "terminus-2"}}}


@pytest.mark.fast
def test_task_batch_adapter_does_not_enter_regular_invoke_pool() -> None:
    manager = _manager()
    adapter = _TaskBatchHarnessAdapter()

    manager.register_role_adapter("harbor", adapter)

    with pytest.raises(KeyError, match="not registered for sample invocation"):
        with manager.borrow_role("harbor"):
            pass
    assert manager.snapshot()["adapters"] == []


@pytest.mark.fast
def test_plain_inference_adapter_still_uses_regular_invoke_pool() -> None:
    manager = _manager()
    adapter = _InvokableAdapter("dut")

    manager.register_role_adapter("dut", adapter)

    with manager.borrow_role("dut") as role:
        result = role.invoke({"prompt": "hello"}, trace=None)

    assert result == {"answer": "hello"}


@pytest.mark.fast
def test_future_sample_step_external_adapter_without_task_batch_capability_uses_regular_pool() -> None:
    manager = _manager()
    adapter = _InvokableAdapter(
        "external_sample",
        role_type="external_harness",
        capabilities=("sample_step_harness",),
    )

    manager.register_role_adapter("external_sample", adapter)

    assert manager.get_task_batch_harness_adapter("external_sample") is None
    with manager.borrow_role("external_sample") as role:
        result = role.invoke({"prompt": "sample"}, trace=None)

    assert result == {"answer": "sample"}


@pytest.mark.fast
def test_sample_step_external_adapter_with_similar_methods_stays_in_regular_pool() -> None:
    manager = _manager()
    adapter = _SampleStepExternalWithTaskBatchMethodNames()

    manager.register_role_adapter("external_sample_structural", adapter)

    assert manager.get_task_batch_harness_adapter("external_sample_structural") is None
    with manager.borrow_role("external_sample_structural") as role:
        result = role.invoke({"prompt": "sample"}, trace=None)

    assert result == {"answer": "sample"}


@pytest.mark.fast
def test_structural_task_batch_protocol_without_role_adapter_is_not_task_batch_managed() -> None:
    manager = _manager()
    adapter = _StructuralTaskBatchHarnessAdapter()

    manager.register_role_adapter("structural", adapter)

    assert manager.get_task_batch_harness_adapter("structural") is None


@pytest.mark.fast
def test_reregistering_normal_adapter_as_task_batch_clears_regular_pool_state() -> None:
    manager = _manager()
    manager.register_role_adapter("switchable", _InvokableAdapter("switchable"))

    manager.register_role_adapter("switchable", _TaskBatchHarnessAdapter("switchable"))

    assert manager.get_task_batch_harness_adapter("switchable") is manager.get_adapter("switchable")
    assert manager.snapshot()["adapters"] == []
    with pytest.raises(KeyError, match="not registered for sample invocation"):
        with manager.borrow_role("switchable"):
            pass


@pytest.mark.fast
def test_reregistering_task_batch_adapter_as_normal_clears_task_batch_registry() -> None:
    manager = _manager()
    manager.register_role_adapter("switchable", _TaskBatchHarnessAdapter("switchable"))

    manager.register_role_adapter("switchable", _InvokableAdapter("switchable"))

    assert manager.get_task_batch_harness_adapter("switchable") is None
    with manager.borrow_role("switchable") as role:
        result = role.invoke({"prompt": "normal"}, trace=None)

    assert result == {"answer": "normal"}
