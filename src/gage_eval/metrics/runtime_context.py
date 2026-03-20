"""Runtime context passed into metric aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gage_eval.evaluation.cache import EvalCache


@dataclass(frozen=True)
class AggregationRuntimeContext:
    """Carry task-scoped runtime details into aggregators.

    Aggregators remain pure Python objects registered in ``MetricRegistry``.
    This context gives them read-only access to run metadata that is only known
    once a task runtime is materialized.
    """

    run_id: str
    task_id: Optional[str]
    run_dir: Optional[Path]
    cache_store: Optional["EvalCache"] = None
    details_namespace: Optional[str] = None
    shuffle_artifact_root: Optional[Path] = None
