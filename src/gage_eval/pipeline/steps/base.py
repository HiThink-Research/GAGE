"""Shared step abstractions."""

from __future__ import annotations

from enum import Enum


class StepKind(str, Enum):
    SAMPLE = "sample"
    GLOBAL = "global"


class Step:
    """Minimal step base class keeping the step kind/name."""

    def __init__(self, name: str, kind: StepKind) -> None:
        self.name = name
        self.kind = kind

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, kind={self.kind.value!r})"


class SampleStep(Step):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, kind=StepKind.SAMPLE)


class GlobalStep(Step):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, kind=StepKind.GLOBAL)
