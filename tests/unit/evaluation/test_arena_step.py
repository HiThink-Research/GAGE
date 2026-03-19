from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.arena import ArenaStep
from gage_eval.reporting.recorders import InMemoryRecorder


class _RoleStub:
    def invoke(self, payload: dict[str, Any], trace: ObservabilityTrace) -> dict[str, Any]:
        _ = payload, trace
        return {"result": "draw"}


class _RoleManagerStub:
    @contextmanager
    def borrow_role(self, adapter_id: str) -> Iterator[_RoleStub]:
        _ = adapter_id
        yield _RoleStub()


def test_arena_step_emits_primary_lifecycle_events_once() -> None:
    trace = ObservabilityTrace(
        recorder=InMemoryRecorder(run_id="arena-step-events"),
        run_id="arena-step-events",
    )
    step = ArenaStep(adapter_id="arena_adapter")

    output = step.execute(
        sample={"id": "sample-1"},
        role_manager=_RoleManagerStub(),
        trace=trace,
    )

    assert output == {"result": "draw"}
    events = [event["event"] for event in trace.events]
    assert events == ["arena_start", "arena_end"]
