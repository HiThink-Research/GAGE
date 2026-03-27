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


class SupportUnit(Protocol):
    def invoke(self, context: SupportContext) -> SupportContext:
        ...
