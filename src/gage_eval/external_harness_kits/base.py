"""External harness adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable


@dataclass(frozen=True)
class TaskBatchHarnessRequest:
    adapter_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskBatchHarnessPlan:
    adapter_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskBatchHarnessHandle:
    adapter_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskBatchHarnessResult:
    adapter_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TaskBatchHarnessAdapter(Protocol):
    adapter_id: str

    def translate(self, request: TaskBatchHarnessRequest) -> TaskBatchHarnessPlan: ...

    def launch(self, plan: TaskBatchHarnessPlan) -> TaskBatchHarnessHandle: ...

    def poll_until_done(self, handle: TaskBatchHarnessHandle) -> TaskBatchHarnessResult: ...

    def parse_results(self, result: TaskBatchHarnessResult) -> Iterable[Any]: ...


__all__ = [
    "TaskBatchHarnessAdapter",
    "TaskBatchHarnessHandle",
    "TaskBatchHarnessPlan",
    "TaskBatchHarnessRequest",
    "TaskBatchHarnessResult",
]
