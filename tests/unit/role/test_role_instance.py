from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from gage_eval.role.role_instance import (
    ConversationHistory,
    HistorySnapshotPolicy,
    Role,
)


class RecordingAdapter:
    """Capture adapter invocations for role instance tests."""

    def __init__(self, result: Optional[Dict[str, Any]] = None) -> None:
        self.received_payload: Optional[Dict[str, Any]] = None
        self.received_state: Optional[Dict[str, Any]] = None
        self._result = result or {"answer": "ok"}

    def clone_for_sample(self) -> Dict[str, Any]:
        return {"state": "sample"}

    def invoke(self, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        self.received_payload = payload
        self.received_state = state
        return self._result


class CountingHistory(ConversationHistory):
    """Track snapshot calls for payload materialization tests."""

    def __init__(self, initial_messages=None) -> None:
        super().__init__(initial_messages)
        self.snapshot_calls = 0

    def snapshot(self):  # type: ignore[override]
        self.snapshot_calls += 1
        return super().snapshot()


@pytest.mark.fast
def test_role_invoke_passes_payload_through_when_no_history_is_needed() -> None:
    adapter = RecordingAdapter()
    role = Role("adapter", adapter)
    payload = {"sample": {"id": "sample-1"}}

    role.invoke(payload, trace=None)

    assert adapter.received_payload is payload
    assert adapter.received_payload == {"sample": {"id": "sample-1"}}


@pytest.mark.fast
def test_role_invoke_reuses_explicit_messages_for_history_without_snapshot() -> None:
    adapter = RecordingAdapter()
    history = CountingHistory([{"role": "system", "content": "seed"}])
    role = Role("adapter", adapter, history=history)
    messages = [{"role": "user", "content": "hello"}]

    role.invoke({"sample": {}, "messages": messages}, trace=None)

    assert history.snapshot_calls == 0
    assert adapter.received_payload is not None
    assert adapter.received_payload["messages"] is messages
    assert adapter.received_payload["history"] is messages


@pytest.mark.fast
def test_role_invoke_reuses_explicit_history_for_messages_without_snapshot() -> None:
    adapter = RecordingAdapter()
    history = CountingHistory([{"role": "system", "content": "seed"}])
    role = Role("adapter", adapter, history=history)
    explicit_history = [{"role": "assistant", "content": "done"}]

    role.invoke({"sample": {}, "history": explicit_history}, trace=None)

    assert history.snapshot_calls == 0
    assert adapter.received_payload is not None
    assert adapter.received_payload["history"] is explicit_history
    assert adapter.received_payload["messages"] is explicit_history


@pytest.mark.fast
def test_role_invoke_injects_internal_history_when_payload_omits_both_views() -> None:
    adapter = RecordingAdapter()
    history = CountingHistory([{"role": "system", "content": "seed"}])
    role = Role("adapter", adapter, history=history)
    payload = {"sample": {"id": "sample-2"}}

    role.invoke(payload, trace=None)

    assert history.snapshot_calls == 1
    assert adapter.received_payload is not payload
    assert adapter.received_payload is not None
    assert adapter.received_payload["history"] == [{"role": "system", "content": "seed"}]
    assert adapter.received_payload["messages"] == [{"role": "system", "content": "seed"}]


@pytest.mark.fast
def test_role_invoke_strips_history_controls_and_persists_cloned_history() -> None:
    adapter = RecordingAdapter(
        result={
            "history_delta": {
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
            }
        }
    )
    role = Role("adapter", adapter)
    override = {"role": "system", "content": "seed"}
    request_delta = {"role": "user", "content": [{"type": "text", "text": "hello"}]}

    role.invoke(
        {
            "sample": {},
            "history_override": override,
            "history_delta": request_delta,
        },
        trace=None,
    )

    assert adapter.received_payload is not None
    assert "history_override" not in adapter.received_payload
    assert "history_delta" not in adapter.received_payload

    override["content"] = "mutated"
    request_delta["content"][0]["text"] = "changed"
    snapshot = role.history_snapshot()
    assert snapshot == [
        {"role": "system", "content": "seed"},
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]


@pytest.mark.fast
def test_conversation_history_snapshot_clones_nested_message_content() -> None:
    history = ConversationHistory(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    )

    snapshot = history.snapshot()
    snapshot[0]["content"][0]["text"] = "changed"

    assert history.snapshot()[0]["content"][0]["text"] == "hello"


@pytest.mark.fast
def test_role_caches_resolved_history_policy() -> None:
    adapter = RecordingAdapter()
    adapter.params = {  # type: ignore[attr-defined]
        "history_policy": {
            "copy_mode": "deep",
            "window_messages": 2,
        }
    }
    role = Role("adapter", adapter)

    first = role._resolve_history_policy()
    second = role._resolve_history_policy()

    assert isinstance(first, HistorySnapshotPolicy)
    assert first is second


@pytest.mark.fast
def test_role_invoke_caches_binding_overlay_but_copies_per_payload() -> None:
    class _RouteDecision:
        def __init__(self) -> None:
            self.runtime_mode = "native"
            self.calls = 0

        def to_payload(self) -> Dict[str, Any]:
            self.calls += 1
            return {"runtime_mode": "native", "route_key": "route-1"}

    class _ExecutionContext:
        run_id = "run-1"
        task_id = "task-1"
        sample_id = "sample-1"
        step_type = "inference"
        adapter_id = "adapter"
        step_slot_id = "slot-1"

    adapter = RecordingAdapter()
    role = Role("adapter", adapter)
    route_decision = _RouteDecision()
    role.attach_invocation_binding(route_decision, execution_context=_ExecutionContext())

    role.invoke({}, trace=None)
    first_payload = adapter.received_payload
    assert first_payload is not None
    first_payload["runtime_route"]["route_key"] = "mutated"
    first_payload["execution_context"]["run_id"] = "changed"

    role.invoke({}, trace=None)
    second_payload = adapter.received_payload

    assert second_payload is not None
    assert route_decision.calls == 1
    assert second_payload["runtime_route"]["route_key"] == "route-1"
    assert second_payload["execution_context"]["run_id"] == "run-1"
