"""Shared step abstractions."""

from __future__ import annotations

from enum import Enum


class StepKind(str, Enum):
    SAMPLE = "sample"
    GLOBAL = "global"


class Step:
    """Minimal step base class keeping the step kind/name."""

    def __init__(self, name: str, kind: StepKind) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "_task_scope_frozen", False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, kind={self.kind.value!r})"

    def __setattr__(self, name: str, value) -> None:
        if name != "_task_scope_frozen" and getattr(self, "_task_scope_frozen", False):
            raise AttributeError(
                f"{self.__class__.__name__} is task-scope frozen; execution state must not be stored on the step instance"
            )
        object.__setattr__(self, name, value)

    def freeze(self) -> None:
        object.__setattr__(self, "_task_scope_frozen", True)

    @property
    def is_frozen(self) -> bool:
        return bool(getattr(self, "_task_scope_frozen", False))


class SampleStep(Step):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, kind=StepKind.SAMPLE)


class GlobalStep(Step):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, kind=StepKind.GLOBAL)
