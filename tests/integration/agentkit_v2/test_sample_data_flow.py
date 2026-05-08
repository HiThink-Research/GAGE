from __future__ import annotations

import pytest

from gage_eval.assets.datasets.sample import Message, MessageContent, Sample
from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.evaluation.task_planner import TaskPlanner
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder
from gage_eval.role.adapters.dut_agent import DUTAgentAdapter
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager


class _RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def aexecute(self, *, sample, payload, trace=None):
        self.calls.append({"sample": sample, "payload": payload, "trace": trace})
        return {
            "answer": "ok",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        }


@pytest.mark.fast
def test_sample_loop_preserves_metadata_for_dut_agent_without_sandbox() -> None:
    sample = Sample(
        schema_version="0.0.1",
        id="instance-1",
        messages=[Message(role="user", content=[MessageContent(type="text", text="fix")])],
        metadata={"image_uri": "jefzda/sweap-images:nodebb"},
    )
    executor = _RecordingExecutor()
    adapter = DUTAgentAdapter(
        adapter_id="dut",
        role_type="dut_agent",
        capabilities=(),
        executor_ref=executor,
    )
    role_manager = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=2)]))
    role_manager.register_role_adapter("dut", adapter)
    planner = TaskPlanner()
    planner.configure_custom_steps([{"step": "inference", "adapter_id": "dut"}])
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="sample-flow"))
    loop = SampleLoop([sample], concurrency=1)

    loop.run(planner, role_manager, trace)

    assert loop.processed_count == 1
    assert len(executor.calls) == 1
    received = executor.calls[0]["sample"]
    assert isinstance(received, dict)
    assert received["id"] == "instance-1"
    assert received["metadata"]["image_uri"] == "jefzda/sweap-images:nodebb"
    assert "sandbox" not in received
    assert "sandbox" not in executor.calls[0]["payload"]["sample"]
