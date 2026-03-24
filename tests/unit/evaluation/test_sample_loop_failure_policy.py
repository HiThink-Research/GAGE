from __future__ import annotations

import threading
import time
from typing import Any, Dict, Iterable

import pytest

from gage_eval.evaluation.execution_controller import SampleLoopExecutionError
from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.evaluation.task_planner import TaskPlanner
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder
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
        return {"mcp_endpoint": "http://fake"}

    def teardown(self) -> None:
        FakeSandbox.teardown_calls += 1


class _FailingInferenceAdapter:
    role_type = "dut_model"
    resource_requirement = {}
    backend = None

    def __init__(
        self,
        fail_on: str | Iterable[str] = "s0",
        *,
        wait_before_failure: threading.Event | None = None,
        wait_before_failure_by_sample: Dict[str, threading.Event] | None = None,
        sleep_by_sample: Dict[str, float] | None = None,
        started_events: Dict[str, threading.Event] | None = None,
    ) -> None:
        self.adapter_id = "dut"
        if isinstance(fail_on, str):
            self.fail_on = {fail_on}
        else:
            self.fail_on = {item for item in fail_on}
        self.sandbox_config = {"runtime": "fake"}
        self.wait_before_failure = wait_before_failure
        self.wait_before_failure_by_sample = wait_before_failure_by_sample or {}
        self.sleep_by_sample = sleep_by_sample or {}
        self.started_events = started_events or {}

    def clone_for_sample(self):
        return self

    def invoke(self, payload, state=None):
        sample = payload.get("sample", {})
        sample_id = sample.get("id")
        started_event = self.started_events.get(sample_id)
        if started_event is not None:
            started_event.set()
        provider = payload.get("sandbox_provider")
        if provider is not None:
            provider.get_handle()
        delay = self.sleep_by_sample.get(sample_id)
        if delay:
            time.sleep(delay)
        if sample_id in self.fail_on:
            wait_event = self.wait_before_failure_by_sample.get(sample_id)
            if wait_event is not None:
                wait_event.wait(timeout=1.0)
            elif self.wait_before_failure is not None:
                self.wait_before_failure.wait(timeout=1.0)
            raise RuntimeError("boom")
        msg = f"ok-{sample_id}"
        return {
            "answer": msg,
            "message": {"role": "assistant", "content": [{"type": "text", "text": msg}]},
        }


class _ThreadCapturingAdapter:
    role_type = "dut_model"
    resource_requirement = {}
    backend = None

    def __init__(self) -> None:
        self.adapter_id = "dut"
        self.thread_names: list[str] = []

    def clone_for_sample(self):
        return self

    def invoke(self, payload, state=None):
        del payload, state
        self.thread_names.append(threading.current_thread().name)
        return {
            "answer": "ok",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        }


def _build_runtime(
    *,
    failure_policy: str,
    sample_count: int = 3,
    concurrency: int = 1,
    max_inflight: int = 2,
    adapter: _FailingInferenceAdapter | None = None,
):
    FakeSandbox.start_calls = 0
    FakeSandbox.teardown_calls = 0
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    samples = [
        {
            "id": f"s{idx}",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "choices": [],
        }
        for idx in range(sample_count)
    ]
    sample_loop = SampleLoop(
        samples,
        concurrency=concurrency,
        max_inflight=max_inflight,
        failure_policy=failure_policy,
        sandbox_manager=manager,
    )
    planner = TaskPlanner()
    planner.configure_custom_steps([{"step": "inference", "adapter_id": "dut"}])
    role_manager = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=1)]))
    role_manager.register_role_adapter("dut", adapter or _FailingInferenceAdapter())
    borrow_calls: list[str | None] = []
    original_borrow = role_manager.borrow_role

    def counting_borrow(adapter_id, *, execution_context=None):
        borrow_calls.append(adapter_id)
        return original_borrow(adapter_id, execution_context=execution_context)

    role_manager.borrow_role = counting_borrow  # type: ignore[assignment]
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id=f"sample-loop-{failure_policy}"))
    return sample_loop, planner, role_manager, trace, borrow_calls


@pytest.mark.fast
def test_fail_fast_cancels_unstarted_samples_without_touching_resources() -> None:
    sample_loop, planner, role_manager, trace, borrow_calls = _build_runtime(
        failure_policy="fail_fast",
        sample_count=3,
    )

    with pytest.raises(SampleLoopExecutionError) as exc_info:
        sample_loop.run(planner, role_manager, trace)

    outcome = exc_info.value.outcome
    sample_loop.shutdown()
    role_manager.shutdown()

    assert outcome.status == "aborted"
    assert outcome.failed_sample_id == "s0"
    assert outcome.cancelled_samples >= 1
    assert FakeSandbox.start_calls == 1
    assert len(borrow_calls) == 1


@pytest.mark.fast
def test_graceful_allows_running_samples_to_finish_after_first_error() -> None:
    sample_started = threading.Event()
    sample_loop, planner, role_manager, trace, borrow_calls = _build_runtime(
        failure_policy="graceful",
        sample_count=3,
        concurrency=2,
        max_inflight=2,
        adapter=_FailingInferenceAdapter(
            wait_before_failure=sample_started,
            sleep_by_sample={"s1": 0.05},
            started_events={"s1": sample_started},
        ),
    )

    with pytest.raises(SampleLoopExecutionError) as exc_info:
        sample_loop.run(planner, role_manager, trace)

    outcome = exc_info.value.outcome
    sample_loop.shutdown()
    role_manager.shutdown()

    assert outcome.status == "aborted"
    assert outcome.completed_after_first_error >= 1
    assert outcome.cancelled_samples >= 1
    assert FakeSandbox.start_calls == 2
    assert len(borrow_calls) == 2


@pytest.mark.fast
def test_best_effort_runs_remaining_samples_after_failure() -> None:
    sample_loop, planner, role_manager, trace, borrow_calls = _build_runtime(
        failure_policy="best_effort",
        sample_count=3,
    )

    with pytest.raises(SampleLoopExecutionError) as exc_info:
        sample_loop.run(planner, role_manager, trace)

    outcome = exc_info.value.outcome
    sample_loop.shutdown()
    role_manager.shutdown()

    assert outcome.status == "completed_with_failures"
    assert outcome.cancelled_samples == 0
    assert FakeSandbox.start_calls == 3
    assert len(borrow_calls) == 3


@pytest.mark.fast
def test_graceful_does_not_count_post_error_failed_samples_as_completed() -> None:
    sample_started = threading.Event()
    sample_loop, planner, role_manager, trace, borrow_calls = _build_runtime(
        failure_policy="graceful",
        sample_count=3,
        concurrency=2,
        max_inflight=2,
        adapter=_FailingInferenceAdapter(
            fail_on={"s0", "s1"},
            wait_before_failure=sample_started,
            sleep_by_sample={"s1": 0.05},
            started_events={"s1": sample_started},
        ),
    )

    with pytest.raises(SampleLoopExecutionError) as exc_info:
        sample_loop.run(planner, role_manager, trace)

    outcome = exc_info.value.outcome
    sample_loop.shutdown()
    role_manager.shutdown()

    assert outcome.status == "aborted"
    assert outcome.failed_sample_id == "s0"
    assert outcome.completed_after_first_error == 0
    assert outcome.cancelled_samples >= 1
    assert FakeSandbox.start_calls == 2
    assert len(borrow_calls) == 2


@pytest.mark.fast
def test_sequential_sample_loop_executes_samples_on_calling_thread() -> None:
    adapter = _ThreadCapturingAdapter()
    sample_loop, planner, role_manager, trace, _borrow_calls = _build_runtime(
        failure_policy="fail_fast",
        sample_count=1,
        concurrency=1,
        adapter=adapter,
    )

    outcome = sample_loop.run(planner, role_manager, trace)
    sample_loop.shutdown()
    role_manager.shutdown()

    assert outcome.status == "completed"
    assert adapter.thread_names == [threading.current_thread().name]
