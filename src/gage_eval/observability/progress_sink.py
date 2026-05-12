"""Progress sink protocol for long-running task-level execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from gage_eval.observability.trace import ObservabilityTrace


@runtime_checkable
class ProgressSink(Protocol):
    def update(
        self,
        *,
        completed: int,
        total: int,
        phase: str,
        elapsed_s: float,
        **extra: Any,
    ) -> None: ...


@dataclass(frozen=True)
class ProgressSnapshot:
    completed: int
    total: int
    phase: str
    elapsed_s: float


class TraceProgressSink:
    """Emit Appendix E progress events while retaining latest counters."""

    def __init__(self, *, trace: ObservabilityTrace, job_name: str) -> None:
        self._trace = trace
        self._job_name = str(job_name)
        self._snapshot = ProgressSnapshot(
            completed=0,
            total=0,
            phase="pending",
            elapsed_s=0.0,
        )

    @property
    def snapshot(self) -> ProgressSnapshot:
        return self._snapshot

    def update(
        self,
        *,
        completed: int,
        total: int,
        phase: str,
        elapsed_s: float,
        **extra: Any,
    ) -> None:
        del extra
        completed_int = _non_negative_int(completed, label="completed")
        total_int = _non_negative_int(total, label="total")
        elapsed = float(elapsed_s)
        self._snapshot = ProgressSnapshot(
            completed=completed_int,
            total=total_int,
            phase=str(phase),
            elapsed_s=elapsed,
        )
        self._trace.emit(
            "external_harness_progress",
            {
                "job_name": self._job_name,
                "completed": completed_int,
                "total": total_int,
                "phase": str(phase),
                "elapsed_s": elapsed,
            },
        )


def _non_negative_int(value: int, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return result


__all__ = ["ProgressSink", "ProgressSnapshot", "TraceProgressSink"]
