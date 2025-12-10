import os
import time

from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.evaluation.task_planner import TaskPlanner
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager


class _EchoAdapter:
    role_type = "dut_model"
    resource_requirement = {}
    backend = None

    def __init__(self, delay_s: float = 0.001) -> None:
        self.delay_s = delay_s
        self.adapter_id = "echo"

    def clone_for_sample(self):
        return self

    def invoke(self, payload, state=None):
        sample = payload.get("sample", {})
        time.sleep(self.delay_s)
        msg = f"echo-{sample.get('id')}"
        return {"answer": msg, "message": {"role": "assistant", "content": [{"type": "text", "text": msg}]}}


def _build_runtime(sample_count: int, *, concurrency: int, prefetch: int, max_inflight: int):
    samples = [
        {
            "id": f"s{idx}",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "choices": [],
        }
        for idx in range(sample_count)
    ]
    loop = SampleLoop(samples, concurrency=concurrency, prefetch_factor=prefetch, max_inflight=max_inflight)
    planner = TaskPlanner()
    planner.configure_custom_steps([{"step": "inference", "adapter_id": "echo"}])
    rm = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=2)]))
    rm.register_role_adapter("echo", _EchoAdapter(delay_s=0.0005))
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="sample-loop-echo"))
    return loop, planner, rm, trace, samples


def test_sample_loop_bounded_buffer():
    loop, planner, rm, trace, samples = _build_runtime(20, concurrency=3, prefetch=2, max_inflight=4)

    loop.run(planner, rm, trace)

    assert loop.processed_count == len(samples)
    assert all(sample.get("predict_result") for sample in samples)

    buffer_events = [e["payload"] for e in trace.events if e["event"] == "sample_buffer_state"]
    assert buffer_events
    cap = max(ev["buffer_capacity"] for ev in buffer_events)
    assert all(ev["buffer_size"] <= cap for ev in buffer_events)
    assert all(ev["inflight"] <= ev["max_inflight"] for ev in buffer_events)

    inf_start = [e for e in trace.events if e["event"] == "inference_start"]
    inf_end = [e for e in trace.events if e["event"] == "inference_end"]
    assert len(inf_start) == len(samples)
    assert len(inf_end) == len(samples)


def test_sample_loop_fire_and_forget():
    loop, planner, rm, trace, samples = _build_runtime(10, concurrency=2, prefetch=1, max_inflight=2)
    os.environ["GAGE_EVAL_FF_MODE"] = "1"
    try:
        loop.run(planner, rm, trace)
    finally:
        os.environ.pop("GAGE_EVAL_FF_MODE", None)

    assert loop.processed_count == len(samples)
    assert all(sample.get("predict_result") for sample in samples)

    inf_events = [e for e in trace.events if e["event"] == "inference_end"]
    assert len(inf_events) == len(samples)
