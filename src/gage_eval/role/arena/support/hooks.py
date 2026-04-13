"""Support workflow lifecycle hook declarations."""

from __future__ import annotations

from enum import Enum
from typing import Protocol

from gage_eval.role.arena.support.context import SupportContext

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        """Python 3.10-compatible subset of enum.StrEnum."""

        def __str__(self) -> str:
            return str(self.value)


class SupportHook(StrEnum):
    AFTER_OBSERVE = "after_observe"
    BEFORE_DECIDE = "before_decide"
    AFTER_DECIDE = "after_decide"
    BEFORE_APPLY = "before_apply"
    AFTER_APPLY = "after_apply"
    ON_FINALIZE = "on_finalize"


class SupportUnitKind(StrEnum):
    PERCEPTION_SUPPORT = "perception_support"
    COGNITIVE_SUPPORT = "cognitive_support"
    EXECUTION_SUPPORT = "execution_support"
    EVIDENCE_SUPPORT = "evidence_support"


class SupportDegradePolicy(StrEnum):
    FAIL_FAST = "fail_fast"
    CONTINUE = "continue"


class SupportUnit(Protocol):
    def invoke(self, context: SupportContext) -> SupportContext:
        ...
