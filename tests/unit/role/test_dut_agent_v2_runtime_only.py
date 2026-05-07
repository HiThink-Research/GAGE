from __future__ import annotations

import asyncio

import pytest

from gage_eval.assets.datasets.sample import Message, MessageContent, Sample
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.adapters.dut_agent import DUTAgentAdapter


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def aexecute(self, *, sample, payload, trace=None):
        self.calls.append({"sample": sample, "payload": payload, "trace": trace})
        return {"answer": "runtime", "sample_id": sample.get("id")}


class FailingLegacyBackend:
    def __init__(self) -> None:
        self.invoke_calls = 0
        self.ainvoke_calls = 0
        self.shutdown_calls = 0

    def invoke(self, payload):
        self.invoke_calls += 1
        raise AssertionError("legacy backend invoke must not be called")

    async def ainvoke(self, payload):
        self.ainvoke_calls += 1
        raise AssertionError("legacy backend ainvoke must not be called")

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        raise AssertionError("legacy backend shutdown must not be called")


class ShutdownTrackingSandboxManager:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.mark.fast
def test_dut_agent_requires_executor_ref() -> None:
    adapter = DUTAgentAdapter(
        adapter_id="dut-runtime",
        role_type="dut_agent",
        capabilities=(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(adapter.ainvoke({"sample": {"id": "sample-1"}}, RoleAdapterState()))

    assert "dut_agent.executor_ref.missing" in str(exc_info.value)
    assert getattr(exc_info.value, "code", None) == "dut_agent.executor_ref.missing"


@pytest.mark.fast
def test_dut_agent_calls_only_executor_ref_even_when_legacy_backend_is_present() -> None:
    executor = RecordingExecutor()
    backend = FailingLegacyBackend()
    adapter = DUTAgentAdapter(
        adapter_id="dut-runtime",
        role_type="dut_agent",
        capabilities=(),
        agent_backend=backend,
        executor_ref=executor,
    )
    trace = object()
    payload = {"sample": {"id": "sample-1"}, "trace": trace, "messages": []}

    result = asyncio.run(adapter.ainvoke(payload, RoleAdapterState()))

    assert result == {"answer": "runtime", "sample_id": "sample-1"}
    assert executor.calls == [{"sample": payload["sample"], "payload": payload, "trace": trace}]
    assert backend.invoke_calls == 0
    assert backend.ainvoke_calls == 0


@pytest.mark.fast
def test_dut_agent_executor_ref_receives_empty_sample_when_payload_omits_sample() -> None:
    executor = RecordingExecutor()
    adapter = DUTAgentAdapter(
        adapter_id="dut-runtime",
        role_type="dut_agent",
        capabilities=(),
        executor_ref=executor,
    )
    payload = {"messages": [{"role": "user", "content": "hi"}]}

    result = asyncio.run(adapter.ainvoke(payload, RoleAdapterState()))

    assert result == {"answer": "runtime", "sample_id": None}
    assert executor.calls == [{"sample": {}, "payload": payload, "trace": None}]


@pytest.mark.fast
def test_dut_agent_normalizes_dataclass_sample_before_executor() -> None:
    executor = RecordingExecutor()
    adapter = DUTAgentAdapter(
        adapter_id="dut-runtime",
        role_type="dut_agent",
        capabilities=(),
        executor_ref=executor,
    )
    sample = Sample(
        schema_version="0.0.1",
        id="instance-1",
        messages=[Message(role="user", content=[MessageContent(type="text", text="fix")])],
        metadata={"image_uri": "jefzda/sweap-images:nodebb"},
    )
    payload = {"sample": sample}

    result = asyncio.run(adapter.ainvoke(payload, RoleAdapterState()))

    assert result == {"answer": "runtime", "sample_id": "instance-1"}
    assert executor.calls[0]["sample"]["metadata"]["image_uri"] == "jefzda/sweap-images:nodebb"
    assert "sandbox" not in executor.calls[0]["sample"]
    assert "sandbox" not in executor.calls[0]["payload"]["sample"]
    assert payload["sample"] is sample


@pytest.mark.fast
def test_dut_agent_shutdown_only_releases_executor_resources() -> None:
    backend = FailingLegacyBackend()
    adapter_sandbox_manager = ShutdownTrackingSandboxManager()
    executor_sandbox_manager = ShutdownTrackingSandboxManager()
    executor_ref = type(
        "_ExecutorRef",
        (),
        {
            "resource_manager": type(
                "_ResourceManager",
                (),
                {"_sandbox_manager": executor_sandbox_manager},
            )()
        },
    )()
    adapter = DUTAgentAdapter(
        adapter_id="dut-runtime",
        role_type="dut_agent",
        capabilities=(),
        agent_backend=backend,
        sandbox_manager=adapter_sandbox_manager,
        executor_ref=executor_ref,
    )

    adapter.shutdown()

    assert executor_sandbox_manager.shutdown_calls == 1
    assert backend.shutdown_calls == 0
    assert adapter_sandbox_manager.shutdown_calls == 0
