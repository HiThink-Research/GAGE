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
from gage_eval.registry import import_kind_from_manifest, registry

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


def _instantiate_summary_generator(name: str, *, registry_view=None) -> Optional[Any]:
    lookup = registry_view or registry
    try:
        obj = lookup.get("summary_generators", name)
    except KeyError:
        return None
    if inspect.isclass(obj):
        return obj()
    if hasattr(obj, "generate"):
        return obj
    if callable(obj):
        return _CallableSummaryGenerator(obj)
    return None


def _ensure_summary_generators_loaded(*, registry_view=None) -> None:
    if registry_view is not None:
        return
    try:
        import_kind_from_manifest("summary_generators", registry=registry)
    except Exception as exc:
        _LOGGER.warning(
            "report",
            "Summary generator auto-discover failed: {}",
            exc,
        )


def _select_summary_entries(cache: EvalCache, *, registry_view=None) -> List[Any]:
    lookup = registry_view or registry
    _ensure_summary_generators_loaded(registry_view=registry_view)
    entries = lookup.list("summary_generators")
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


def _build_runtime_health(cache: EvalCache) -> Dict[str, Any]:
    records = list(cache.iter_samples())
    health = {
        "sample_count": len(records),
        "completed_count": 0,
        "failed_count": 0,
        "aborted_count": 0,
        "verifier_skipped_count": 0,
        "scheduler_failed_count": 0,
    }
    for record in records:
        judge_output = _record_judge_output(record)
        scheduler_result = _record_scheduler_result(record)
        runtime_failure = _record_runtime_failure(record)
        scheduler_failed = _scheduler_failed(scheduler_result, runtime_failure)
        verifier_skipped = _verifier_skipped(judge_output)
        status = str(record.get("status") or judge_output.get("status") or "")

        if scheduler_failed:
            health["scheduler_failed_count"] += 1
        if verifier_skipped:
            health["verifier_skipped_count"] += 1
        if status == "aborted":
            health["aborted_count"] += 1
        elif scheduler_failed or status == "failed":
            health["failed_count"] += 1
        elif status == "completed":
            health["completed_count"] += 1
    return health


def _record_judge_output(record: Dict[str, Any]) -> Dict[str, Any]:
    judge_output = record.get("judge_output")
    if isinstance(judge_output, dict):
        return dict(judge_output)
    model_output = record.get("model_output")
    if not isinstance(model_output, dict):
        return {}
    runtime_outcome = model_output.get("runtime_judge_outcome")
    if isinstance(runtime_outcome, dict) and isinstance(runtime_outcome.get("judge_output"), dict):
        return dict(runtime_outcome["judge_output"])
    return {}


def _record_scheduler_result(record: Dict[str, Any]) -> Dict[str, Any]:
    model_output = record.get("model_output")
    if not isinstance(model_output, dict):
        return {}
    runtime_outcome = model_output.get("runtime_judge_outcome")
    if not isinstance(runtime_outcome, dict):
        return {}
    verifier_input = runtime_outcome.get("verifier_input")
    if isinstance(verifier_input, dict) and isinstance(verifier_input.get("scheduler_result"), dict):
        return dict(verifier_input["scheduler_result"])
    return {}


def _record_runtime_failure(record: Dict[str, Any]) -> Dict[str, Any]:
    model_output = record.get("model_output")
    if isinstance(model_output, dict) and isinstance(model_output.get("runtime_failure"), dict):
        return dict(model_output["runtime_failure"])
    return {}


def _scheduler_failed(scheduler_result: Dict[str, Any], runtime_failure: Dict[str, Any]) -> bool:
    if scheduler_result.get("status") in {"failed", "aborted"}:
        return True
    failure_code = str(
        runtime_failure.get("failure_code")
        or scheduler_result.get("failure_code")
        or ""
    )
    return failure_code.startswith("client_execution.")


def _verifier_skipped(judge_output: Dict[str, Any]) -> bool:
    if judge_output.get("status") == "skipped":
        return True
    return judge_output.get("failure_code") == "verifier.skipped_due_to_scheduler_failure"


@registry.asset(
    "pipeline_steps",
    "report",
    desc="Global step that aggregates evaluation results and writes reports",
    tags=("report",),
    step_kind="global",
)
class ReportStep(GlobalStep):
    def __init__(
        self,
        auto_eval_step: Optional["AutoEvalStep"],
        cache_store: EvalCache,
        *,
        registry_view=None,
    ) -> None:
        super().__init__("ReportStep")
        self._auto_eval_step = auto_eval_step
        self._cache = cache_store
        self._registry_view = registry_view

    def get_sample_count(self) -> int:
        """Return the number of persisted samples without leaking cache internals."""

        return int(self._cache.sample_count)

    @property
    def cache_store(self) -> EvalCache:
        return self._cache

    def record_timing(self, phase: str, seconds: float) -> None:
        self._cache.record_timing(phase, seconds)

    def get_timing(self, phase: str) -> Optional[float]:
        return self._cache.get_timing(phase)

    def record_execution_summary(self, payload: Dict[str, Any]) -> None:
        """Persist runtime execution summary for later report merging."""

        self._cache.set_metadata("execution_summary", dict(payload))

    def _collect_summary_payload(self) -> Dict[str, Any]:
        entries = _select_summary_entries(self._cache, registry_view=self._registry_view)
        payload: Dict[str, Any] = {}
        for entry in entries:
            generator = _instantiate_summary_generator(entry.name, registry_view=self._registry_view)
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
        payload_fn=lambda self, *args, **kwargs: {"sample_count": self.get_sample_count()},
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
            "sample_count": self.get_sample_count(),
            "runtime_health": _build_runtime_health(self._cache),
        }
        execution_summary = self._cache.get_metadata("execution_summary")
        if isinstance(execution_summary, dict):
            payload["execution"] = execution_summary
        validation_summary = self._cache.get_metadata("validation_summary")
        if isinstance(validation_summary, dict):
            payload.update(validation_summary)
        if summary_payload:
            payload.update(summary_payload)
        if formatted_tasks:
            payload["tasks"] = formatted_tasks
        payload.update(trace.health_snapshot())
        self._cache.write_summary(payload)
        _LOGGER.info(
            "report",
            "ReportStep finalized summary (samples={}, metrics={}, tasks={})",
            self.get_sample_count(),
            len(formatted_metrics),
            len(formatted_tasks or []),
            trace=trace,
        )
        trace.emit("report_finalize", payload)
        return payload
