"""Action shaping support units."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any

from gage_eval.role.arena.support.context import SupportContext


@dataclass(frozen=True)
class ContinuousActionShapingUnit:
    low: float
    high: float

    def invoke(self, context: SupportContext | dict[str, Any]) -> SupportContext | dict[str, Any]:
        if isinstance(context, SupportContext):
            action = context.payload.get("action")
            if action is not None:
                shaped = self._try_shape_action(action)
                if shaped is not None:
                    context.payload["action"] = shaped
            return context

        action = context.get("action")
        if action is None:
            return dict(context)

        result = dict(context)
        shaped = self._try_shape_action(action)
        if shaped is not None:
            result["action"] = shaped
        return result

    def _clamp(self, value: float) -> float:
        return max(self.low, min(self.high, value))

    def _try_shape_action(self, action: Any) -> list[float] | None:
        try:
            return self._shape_action(action)
        except (TypeError, ValueError):
            return None

    def _shape_action(self, action: Any) -> list[float]:
        if isinstance(action, Iterable) and not isinstance(action, (str, bytes)):
            return [self._clamp(float(value)) for value in action]
        return [self._clamp(float(action))]
