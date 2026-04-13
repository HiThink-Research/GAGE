from __future__ import annotations

from gage_eval.role.arena.support.context import SupportContext
from gage_eval.role.arena.support.hooks import SupportHook
from gage_eval.role.arena.support.workflow import GameSupportWorkflow


class _RecordingUnit:
    def __init__(self, label: str) -> None:
        self.label = label

    def invoke(self, context: SupportContext) -> SupportContext:
        context.unit_trace.append(self.label)
        return context


def test_support_workflow_invokes_units_in_declared_order() -> None:
    workflow = GameSupportWorkflow(
        workflow_id="ordered",
        units_by_hook={
            SupportHook.BEFORE_APPLY: [
                _RecordingUnit("first"),
                _RecordingUnit("second"),
                _RecordingUnit("third"),
            ]
        },
    )

    context = SupportContext()
    result = workflow.run(SupportHook.BEFORE_APPLY, context)

    assert result.unit_trace == ["first", "second", "third"]
