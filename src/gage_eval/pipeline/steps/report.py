"""Global ReportStep that summarizes cached auto-eval results."""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

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


class _CallableSummaryGenerator:
    def __init__(self, func: Callable[[EvalCache], Any]) -> None:
        self._func = func

    def generate(self, cache: EvalCache) -> Any:
        return self._func(cache)


def _resolve_requested_generators(cache: EvalCache) -> Optional[Set[str]]:
    raw = cache.get_metadata("summary_generators")
    if raw is None:
        raw = os.environ.get("GAGE_EVAL_SUMMARY_GENERATORS")
    if raw is None:
        return None
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
        return set(items)
    if isinstance(raw, (list, tuple, set)):
        return {str(item).strip() for item in raw if str(item).strip()}
    return None


def _instantiate_summary_generator(name: str) -> Optional[Any]:
    try:
        obj = registry.get("summary_generators", name)
    except KeyError:
        return None
    if inspect.isclass(obj):
        return obj()
    if hasattr(obj, "generate"):
        return obj
    if callable(obj):
        return _CallableSummaryGenerator(obj)
    return None


def _ensure_summary_generators_loaded() -> None:
    try:
        registry.auto_discover("summary_generators", "gage_eval.reporting.summary_generators")
    except Exception as exc:
        _LOGGER.warning(
            "report",
            "Summary generator auto-discover failed: {}",
            exc,
        )


def _select_summary_entries(cache: EvalCache) -> List[Any]:
    _ensure_summary_generators_loaded()
    entries = registry.list("summary_generators")
    default_entries = [entry for entry in entries if entry.extra.get("default_enabled")]
    requested = _resolve_requested_generators(cache)
    if not requested:
        return default_entries
    requested_set = set(requested)
    requested_entries = [entry for entry in entries if entry.name in requested_set]
    missing = sorted(requested_set - {entry.name for entry in requested_entries})
    if missing:
        _LOGGER.warning(
            "report",
            "Summary generators not found: {}",
            ", ".join(missing),
        )
    selected = {entry.name: entry for entry in default_entries}
    for entry in requested_entries:
        selected.setdefault(entry.name, entry)
    return list(selected.values())


@registry.asset(
    "pipeline_steps",
    "report",
    desc="Global step that aggregates evaluation results and writes reports",
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

    def _collect_summary_payload(self) -> Dict[str, Any]:
        entries = _select_summary_entries(self._cache)
        payload: Dict[str, Any] = {}
        for entry in entries:
            generator = _instantiate_summary_generator(entry.name)
            if generator is None:
                _LOGGER.warning("report", "Summary generator '{}' could not be instantiated", entry.name)
                continue
            try:
                summary = generator.generate(self._cache)
            except Exception as exc:
                _LOGGER.warning(
                    "report",
                    "Summary generator '{}' failed: {}",
                    entry.name,
                    exc,
                )
                continue
            if not summary:
                continue
            if not isinstance(summary, dict):
                _LOGGER.warning(
                    "report",
                    "Summary generator '{}' returned non-dict payload",
                    entry.name,
                )
                continue
            for key, value in summary.items():
                if key in payload:
                    _LOGGER.warning(
                        "report",
                        "Summary key '{}' duplicated; keeping first payload",
                        key,
                    )
                    continue
                payload[key] = value
        return payload

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
        summary_payload = self._collect_summary_payload()
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
        if summary_payload:
            payload.update(summary_payload)
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
