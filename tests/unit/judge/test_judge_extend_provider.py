from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from gage_eval.registry import registry
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.adapters.judge_extend import JudgeExtendAdapter


class FakeProvider:
    def runtime_handle(self) -> dict[str, str]:
        return {"env_endpoint": "http://env"}


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
    provider = FakeProvider()
    adapter = JudgeExtendAdapter(adapter_id="judge_extend_main", implementation="dummy_provider_impl")
    result = asyncio.run(adapter.ainvoke({"sandbox_provider": provider}, RoleAdapterState()))

    assert result["runtime_handle"] == {"env_endpoint": "http://env"}
