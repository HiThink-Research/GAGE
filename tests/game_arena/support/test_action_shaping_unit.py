from __future__ import annotations

from gage_eval.role.arena.support.context import SupportContext
from gage_eval.role.arena.support.hooks import SupportHook
from gage_eval.role.arena.support.workflow import GameSupportWorkflow
from gage_eval.role.arena.support.units.action_shaping import ContinuousActionShapingUnit


def test_continuous_action_shaping_unit_clamps_action_values() -> None:
    unit = ContinuousActionShapingUnit(low=-1.0, high=1.0)

    result = unit.invoke({"action": [3.0, -2.5]})

    assert result == {"action": [1.0, -1.0]}


def test_continuous_action_shaping_unit_handles_iterables_and_preserves_fields() -> None:
    unit = ContinuousActionShapingUnit(low=0.0, high=2.0)

    result = unit.invoke({"action": (1.5, -0.5, 3.0), "meta": {"source": "test"}})

    assert result == {"action": [1.5, 0.0, 2.0], "meta": {"source": "test"}}


def test_continuous_action_shaping_unit_clamps_support_context_payload_via_workflow() -> None:
    workflow = GameSupportWorkflow(
        workflow_id="action-shaping",
        units_by_hook={
            SupportHook.BEFORE_APPLY: [ContinuousActionShapingUnit(low=-1.0, high=1.0)]
        },
    )
    context = SupportContext(payload={"action": [3.0, -2.5]}, state={"stage": "apply"})

    result = workflow.run(SupportHook.BEFORE_APPLY, context)

    assert result is context
    assert result.payload["action"] == [1.0, -1.0]
    assert result.state == {"stage": "apply"}
