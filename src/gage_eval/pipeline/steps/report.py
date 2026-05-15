"""Global ReportStep that summarizes cached auto-eval results."""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from gage_eval.reporting.assembly.context_builder import ReportContextBuilder
from gage_eval.reporting.assembly.extension_runner import SummaryExtensionRunner
from gage_eval.reporting.assembly.health_collector import RuntimeHealthCollector
from gage_eval.reporting.assembly.metric_collector import MetricSummaryCollector
from gage_eval.reporting.assembly.scenario_profiles import ScenarioProfileBuilder
from gage_eval.reporting.contracts import SummaryGeneratorResult
from gage_eval.reporting.evidence.consistency_checker import RunLayoutConsistencyChecker
from gage_eval.reporting.evidence.reader import ReportEvidenceReader
from gage_eval.reporting.persistence.pack_builder import ReportPackBuilder
from gage_eval.reporting.persistence.summary_writer import SummaryWriter
from gage_eval.reporting.privacy import SecretFilter
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
    return RuntimeHealthCollector().collect(records)


def _report_pack_enabled() -> bool:
    raw = os.environ.get("GAGE_EVAL_REPORT_PACK")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _to_summary_payload(result: SummaryGeneratorResult) -> Dict[str, Any]:
    return dict(result.legacy_payload)


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

    def _collect_summary_result(self, context: Dict[str, Any]) -> SummaryGeneratorResult:
        entries = _select_summary_entries(self._cache, registry_view=self._registry_view)
        generators: list[Any] = []
        for entry in entries:
            generator = _instantiate_summary_generator(entry.name, registry_view=self._registry_view)
            if generator is None:
                _LOGGER.warning("report", "Summary generator '{}' could not be instantiated", entry.name)
                continue
            generators.append(generator)
        return SummaryExtensionRunner().run(generators, context)

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
        if pre_write_hook:
            pre_write_hook()
        formatted_metrics = _format_metric_entries(metrics)
        records = list(self._cache.iter_samples())
        collected_metrics = MetricSummaryCollector().collect(formatted_metrics)
        formatted_tasks: Optional[List[Dict[str, Any]]] = None
        if tasks:
            formatted_tasks = []
            for task in tasks:
                task_payload = dict(task)
                task_metrics = task_payload.get("metrics")
                if task_metrics:
                    task_payload["metrics"] = _format_metric_entries(task_metrics)
                formatted_tasks.append(task_payload)
        runtime_health = RuntimeHealthCollector().collect(records)
        payload = {
            "run": self._cache.snapshot(),
            "metrics": formatted_metrics,
            "sample_count": self.get_sample_count(),
            "runtime_health": runtime_health,
        }
        execution_summary = self._cache.get_metadata("execution_summary")
        if isinstance(execution_summary, dict):
            payload["execution"] = execution_summary
        validation_summary = self._cache.get_metadata("validation_summary")
        if isinstance(validation_summary, dict):
            payload.update(validation_summary)
        if formatted_tasks:
            payload["tasks"] = formatted_tasks
        payload.update(trace.health_snapshot())
        index = ReportEvidenceReader().build_index(self._cache.run_dir)
        if (self._cache.run_dir / "summary.json").exists():
            layout_diagnostics = RunLayoutConsistencyChecker().check(self._cache.run_dir)
            index.diagnostics.extend(layout_diagnostics)
        scenario_profiles, scenario_diagnostics = ScenarioProfileBuilder().build(index)
        index.diagnostics.warnings.extend(scenario_diagnostics.get("warnings", []))
        index.diagnostics.errors.extend(scenario_diagnostics.get("errors", []))
        summary_context = {
            "run": payload["run"],
            "samples": records,
            "summary": payload,
            "metrics": collected_metrics,
            "tasks": formatted_tasks or [],
            "evidence_refs": [
                ref.to_dict() if hasattr(ref, "to_dict") else dict(ref)
                for ref in (
                    index.evidence_refs.values()
                    if isinstance(index.evidence_refs, dict)
                    else index.evidence_refs
                )
            ],
        }
        summary_result = self._collect_summary_result(summary_context)
        if summary_result.legacy_payload:
            payload.update(_to_summary_payload(summary_result))
        observability_health = {
            key: payload[key]
            for key in (
                "observability_degraded",
                "observability_mode",
                "backlog_events",
                "events_emitted_total",
                "events_retained_in_memory",
                "events_dropped_by_ring_buffer",
                "events_flushed_total",
            )
            if key in payload
        }
        report_context = ReportContextBuilder().build(
            index=index,
            summary_payload=payload,
            metrics=collected_metrics,
            tasks=formatted_tasks or [],
            runtime_health=runtime_health,
            observability_health=observability_health,
            generator_result=summary_result,
        )
        report_context.scenario_profiles = scenario_profiles
        redaction_result = SecretFilter().redact(records)
        if redaction_result.redacted:
            report_context.diagnostics = dict(report_context.diagnostics or {})
            warnings = list(report_context.diagnostics.get("warnings", []))
            warnings.append(
                {
                    "code": "report_pack.sample_payload_redacted",
                    "redaction_marker": "<redacted:auth>",
                    "finding_count": len(redaction_result.findings),
                }
            )
            report_context.diagnostics["warnings"] = warnings
        report_pack_diagnostics: Dict[str, Any] | None = None
        if _report_pack_enabled():
            try:
                report_pack_diagnostics = ReportPackBuilder().write(
                    self._cache.run_dir,
                    report_context,
                    enabled=True,
                )
                payload["report_pack"] = {
                    "status": report_pack_diagnostics.get("report_pack_status", "completed"),
                    "diagnostics": dict(report_pack_diagnostics),
                }
                trace.emit(
                    "report_pack_generated",
                    {
                        "report_pack_path": report_pack_diagnostics.get("report_pack_path"),
                        "report_pack_status": report_pack_diagnostics.get("report_pack_status", "completed"),
                    },
                )
            except Exception as exc:
                report_pack_diagnostics = {
                    "report_pack_status": "failed",
                    "failure_code": "report_pack.write_failed",
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                }
                payload["report_pack"] = {
                    "status": "failed",
                    "diagnostics": dict(report_pack_diagnostics),
                }
                trace.emit(
                    "report_pack_failed",
                    {
                        "failure_code": "report_pack.write_failed",
                        "report_pack_status": "failed",
                    },
                )
        SummaryWriter().write(
            self._cache,
            payload,
            report_pack_diagnostics=report_pack_diagnostics,
        )
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
