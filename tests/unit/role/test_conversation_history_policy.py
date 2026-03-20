from __future__ import annotations

import pytest

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.role_instance import ConversationHistory, HistorySnapshotPolicy, Role


@pytest.mark.fast
def test_history_snapshot_window_preserves_system_messages() -> None:
    history = ConversationHistory(
        [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "u1"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
            {"role": "user", "content": [{"type": "text", "text": "u2"}]},
        ]
    )

    snapshot = history.snapshot(
        HistorySnapshotPolicy(copy_mode="shallow", window_messages=1)
    )

    assert [item["role"] for item in snapshot] == ["system", "user"]
    assert snapshot[-1]["content"][0]["text"] == "u2"


@pytest.mark.fast
def test_history_snapshot_deep_copy_isolates_nested_mutation() -> None:
    history = ConversationHistory(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
                "tool_calls": [{"id": "call-1", "name": "demo"}],
            }
        ]
    )

    snapshot = history.snapshot(HistorySnapshotPolicy(copy_mode="deep"))
    snapshot[0]["content"][0]["text"] = "changed"
    snapshot[0]["tool_calls"][0]["name"] = "mutated"

    original = history.snapshot(HistorySnapshotPolicy(copy_mode="deep"))
    assert original[0]["content"][0]["text"] == "hello"
    assert original[0]["tool_calls"][0]["name"] == "demo"


@pytest.mark.fast
def test_role_invoke_uses_adapter_history_policy() -> None:
    class CapturingAdapter:
        def __init__(self) -> None:
            self.params = {
                "history_policy": {
                    "copy_mode": "deep",
                    "window_messages": 1,
                    "preserve_system_messages": True,
                }
            }
            self.last_payload = None

        def clone_for_sample(self):
            return object()

        def invoke(self, payload, state):
            self.last_payload = payload
            return {}

    adapter = CapturingAdapter()
    role = Role("dut", adapter)
    role.prime_history(
        [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "u1"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
            {"role": "user", "content": [{"type": "text", "text": "u2"}]},
        ],
        overwrite=True,
    )

    role.invoke({}, ObservabilityTrace())

    assert adapter.last_payload is not None
    assert [item["role"] for item in adapter.last_payload["history"]] == ["system", "user"]
    assert adapter.last_payload["history"][-1]["content"][0]["text"] == "u2"
