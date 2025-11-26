"""Global ReportStep that summarizes cached auto-eval results."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from gage_eval.observability.decorators import observable_stage
from gage_eval.observability.logger import ObservableLogger
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.evaluation.cache import EvalCache
from gage_eval.pipeline.steps.base import GlobalStep
from gage_eval.registry import registry

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from gage_eval.pipeline.steps.auto_eval import AutoEvalStep


_LOGGER = ObservableLogger()


def _format_metric_entries(metrics: Optional[list], decimals: int = 5) -> List[Dict[str, Any]]:
    """Return a copy of metric entries with values rendered to fixed decimals."""

    if not metrics:
        return []
    formatted: List[Dict[str, Any]] = []
    for entry in metrics:
        metric_copy = dict(entry)
        values = metric_copy.get("values")
        if isinstance(values, dict):
            formatted_values: Dict[str, Any] = {}
            raw_values: Dict[str, float] = {}
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    numeric = float(value)
                    formatted_values[key] = f"{numeric:.{decimals}f}"
                    raw_values[key] = numeric
                else:
                    formatted_values[key] = value
            metric_copy["values"] = formatted_values
            if raw_values:
                metric_copy["raw_values"] = raw_values
        formatted.append(metric_copy)
    return formatted


@registry.asset(
    "pipeline_steps",
    "report",
    desc="汇总评测结果并输出报告的阶段",
    tags=("report",),
    step_kind="global",
)
class ReportStep(GlobalStep):
    def __init__(self, auto_eval_step: Optional["AutoEvalStep"], cache_store: EvalCache) -> None:
        super().__init__("ReportStep")
        self._auto_eval_step = auto_eval_step
        self._cache = cache_store

    def record_timing(self, phase: str, seconds: float) -> None:
        self._cache.record_timing(phase, seconds)

    def get_timing(self, phase: str) -> Optional[float]:
        return self._cache.get_timing(phase)

    @observable_stage(
        "report",
        payload_fn=lambda self, *args, **kwargs: {"sample_count": self._cache.sample_count},
    )
    def finalize(
        self,
        trace: ObservabilityTrace,
        *,
        metrics: Optional[list] = None,
        tasks: Optional[list] = None,
        pre_write_hook: Optional[Callable[[], None]] = None,
    ) -> Dict:
        if metrics is None:
            metrics = self._auto_eval_step.aggregated_metrics() if self._auto_eval_step else []
        if pre_write_hook:
            pre_write_hook()
        formatted_metrics = _format_metric_entries(metrics)
        formatted_tasks: Optional[List[Dict[str, Any]]] = None
        if tasks:
            formatted_tasks = []
            for task in tasks:
                task_payload = dict(task)
                task_metrics = task_payload.get("metrics")
                if task_metrics:
                    task_payload["metrics"] = _format_metric_entries(task_metrics)
                formatted_tasks.append(task_payload)
        payload = {
            "run": self._cache.snapshot(),
            "metrics": formatted_metrics,
            "sample_count": self._cache.sample_count,
        }
        if formatted_tasks:
            payload["tasks"] = formatted_tasks
        self._cache.write_summary(payload)
        _LOGGER.info(
            "report",
            "ReportStep finalized summary (samples={}, metrics={}, tasks={})",
            self._cache.sample_count,
            len(formatted_metrics),
            len(formatted_tasks or []),
            trace=trace,
        )
        trace.emit("report_finalize", payload)
        return payload
