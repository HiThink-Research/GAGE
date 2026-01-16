from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from gage_eval.registry import registry
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.adapters.judge_extend import JudgeExtendAdapter
from gage_eval.sandbox.manager import SandboxManager
from gage_eval.sandbox.provider import SandboxProvider, SandboxScope


class FakeSandbox:
    def __init__(self, runtime_configs: Dict[str, Any] | None = None, resources: Dict[str, Any] | None = None):
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}

    def start(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {"env_endpoint": "http://env"}

    def teardown(self) -> None:
        return None


class DummyJudgeImpl:
    def __init__(self, **_: Any) -> None:
        pass

    def invoke(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:
        return {"runtime_handle": payload.get("runtime_handle")}


@pytest.mark.fast
def test_judge_extend_injects_runtime_handle() -> None:
    registry.register(
        "judge_impls",
        "dummy_provider_impl",
        DummyJudgeImpl,
        desc="Dummy judge provider test",
    )
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    provider = SandboxProvider(
        manager,
        {"runtime": "fake"},
        SandboxScope(run_id="run", task_id="task", sample_id="sample"),
    )
    adapter = JudgeExtendAdapter(adapter_id="judge_extend_main", implementation="dummy_provider_impl")
    result = asyncio.run(adapter.ainvoke({"sandbox_provider": provider}, RoleAdapterState()))

    assert result["runtime_handle"] == {"env_endpoint": "http://env"}
    provider.release()
