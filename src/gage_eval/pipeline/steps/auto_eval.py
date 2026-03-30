"""Auto evaluation step that records per-sample metric values."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import as_completed
from typing import Dict, List, Optional, Sequence, Tuple

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.evaluation.execution_controller import TaskExecutionController
from gage_eval.observability.decorators import observable_stage
from gage_eval.observability.logger import ObservableLogger
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.evaluation.cache import EvalCache
from gage_eval.metrics import MetricRegistry, MetricInstance, MetricContext
from gage_eval.metrics.runtime_context import AggregationRuntimeContext
from gage_eval.evaluation.sample_envelope import (
    resolve_arena_trace,
    resolve_judge_output,
    resolve_model_output,
    snapshot_sample,
)
from gage_eval.pipeline.steps.base import SampleStep
from gage_eval.registry import registry


def _auto_eval_sample_id(*args, **kwargs) -> Optional[str]:
    if "sample_id" in kwargs and kwargs["sample_id"] is not None:
        return str(kwargs["sample_id"])
    if len(args) >= 2:
        return str(args[1])
    return None


@registry.asset(
    "pipeline_steps",
    "auto_eval",
    desc="Pipeline step that computes automatic metrics",
    tags=("metrics",),
    step_kind="sample",
)
class AutoEvalStep(SampleStep):
    """Executes automatic metrics for every sample."""

    def __init__(
        self,
        metric_specs: Sequence[MetricSpec],
        metric_registry: Optional[MetricRegistry] = None,
        cache_store: Optional[EvalCache] = None,
        execution_controller: Optional[TaskExecutionController] = None,
    ) -> None:
        super().__init__("AutoEvalStep")
        self._registry = metric_registry or MetricRegistry()
        self._metric_specs: Tuple[MetricSpec, ...] = tuple(metric_specs)
        self._aggregation_runtime_context: Optional[AggregationRuntimeContext] = None
        self._instances: List[MetricInstance] = self._build_metric_instances()
        self._cache = cache_store
        self._aggregate_lock = threading.Lock()
        self._logger = ObservableLogger()
        self._worker_hint = _env_int("GAGE_EVAL_AUTOEVAL_WORKERS")
        self._execution_controller = execution_controller

    def has_metrics(self) -> bool:
        return bool(self._instances)

    def metric_count(self) -> int:
        return len(self._instances)

    def attach_execution_controller(
        self, controller: Optional[TaskExecutionController]
    ) -> None:
        self._execution_controller = controller

    def bind_aggregation_runtime_context(
        self,
        runtime_context: Optional[AggregationRuntimeContext],
    ) -> None:
        self._aggregation_runtime_context = runtime_context
        self._registry.set_runtime_context(runtime_context)
        with self._aggregate_lock:
            self._instances = self._build_metric_instances(runtime_context)

    @observable_stage(
        "auto_eval",
        sample_id_getter=_auto_eval_sample_id,
        payload_fn=lambda self, *args, **kwargs: {"metric_count": len(self._instances)},
    )
    def execute(
        self,
        sample_id: str,
        sample: dict,
        model_output: Dict,
        judge_output: Dict,
        trace: ObservabilityTrace,
        task_id: Optional[str] = None,
    ) -> None:
        resolved_model_output, resolved_judge_output, _ = self._resolve_sample_outputs(
            sample=sample,
            model_output=model_output,
            judge_output=judge_output,
        )
        per_metric_results: Dict[str, Dict] = {}
        logger = self._logger
        worker_count = self._determine_worker_count()
        if self._instances:
            if self._execution_controller is None:
                for instance in self._instances:
                    metric_id, result_dict = self._evaluate_metric(
                        instance,
                        sample_id,
                        sample,
                        resolved_model_output,
                        resolved_judge_output,
                        trace,
                    )
                    per_metric_results[metric_id] = result_dict
            else:
                per_metric_results.update(
                    self._evaluate_metrics_concurrently(
                        sample_id,
                        sample,
                        resolved_model_output,
                        resolved_judge_output,
                        trace,
                    )
                )
        self.persist_sample_artifact(
            sample_id=sample_id,
            sample=sample,
            model_output=resolved_model_output,
            judge_output=resolved_judge_output,
            trace=trace,
            task_id=task_id,
            metrics=per_metric_results,
        )
        if per_metric_results:
            trace.emit(
                "auto_eval_sample",
                payload={
                    "sample_id": sample_id,
                    "task_id": task_id,
                    "metrics": per_metric_results,
                    "worker_count": worker_count,
                },
            )

    def aggregated_metrics(self) -> List[Dict]:
        if not self._instances:
            return []
        with self._aggregate_lock:
            results = [instance.finalize() for instance in self._instances]
        self._logger.info("auto_eval", "Aggregated {} metrics", len(results))
        return results

    def persist_sample_artifact(
        self,
        *,
        sample_id: str,
        sample: dict,
        model_output: Dict,
        judge_output: Dict,
        trace: ObservabilityTrace,
        task_id: Optional[str] = None,
        metrics: Optional[Dict[str, Dict]] = None,
    ) -> None:
        """Persist the canonical per-sample artifact to the cache store."""

        if self._cache is None:
            return
        resolved_model_output, resolved_judge_output, resolved_arena_trace = self._resolve_sample_outputs(
            sample=sample,
            model_output=model_output,
            judge_output=judge_output,
        )
        record = {
            "task_id": task_id,
            "sample": snapshot_sample(sample),
            "model_output": resolved_model_output,
            "arena_trace": resolved_arena_trace,
            "judge_output": resolved_judge_output,
            "metrics": dict(metrics or {}),
        }
        cache_id = self._compose_cache_id(task_id, sample_id)
        namespace = self._resolve_cache_namespace(task_id, resolved_judge_output)
        self._cache.write_sample(cache_id, record, namespace=namespace)
        self._logger.debug(
            "auto_eval",
            "Cached sample artifact sample_id={} task_id={}",
            sample_id,
            task_id,
            trace=trace,
            sample_id=sample_id,
        )

    def _build_metric_instances(
        self,
        runtime_context: Optional[AggregationRuntimeContext] = None,
    ) -> List[MetricInstance]:
        return [
            self._registry.build_metric(spec, runtime_context=runtime_context)
            for spec in self._metric_specs
        ]

    def _evaluate_metric(
        self,
        instance: MetricInstance,
        sample_id: str,
        sample: dict,
        model_output: Dict,
        judge_output: Dict,
        trace: ObservabilityTrace,
    ) -> Tuple[str, Dict]:
        logger = self._logger
        logger.debug(
            "auto_eval",
            "Evaluating metric '{}' for sample_id={}",
            instance.spec.metric_id,
            sample_id,
            trace=trace,
            sample_id=sample_id,
        )
        context = MetricContext(
            sample_id=sample_id,
            sample=sample,
            model_output=model_output,
            judge_output=judge_output,
            args=instance.spec.params,
            trace=trace,
        )
        result = instance.evaluate(context)
        return instance.spec.metric_id, result.to_dict()

    def _evaluate_metrics_concurrently(
        self,
        sample_id: str,
        sample: dict,
        model_output: Dict,
        judge_output: Dict,
        trace: ObservabilityTrace,
    ) -> Dict[str, Dict]:
        per_metric_results: Dict[str, Dict] = {}
        start = time.perf_counter()
        controller = self._execution_controller
        if controller is None:
            return per_metric_results
        future_map = {
            controller.submit_metric(
                self._evaluate_metric,
                instance,
                sample_id,
                sample,
                model_output,
                judge_output,
                trace,
                sample_id=sample_id,
                trace=trace,
            ): instance.spec.metric_id
            for instance in self._instances
        }
        for future in as_completed(future_map):
            metric_id, result_dict = future.result()
            per_metric_results[metric_id] = result_dict
        worker_count = self._determine_worker_count()
        elapsed = time.perf_counter() - start
        self._logger.debug(
            "auto_eval",
            "Concurrent auto-eval completed sample_id={} workers={} elapsed={:.3f}s",
            sample_id,
            worker_count,
            elapsed,
            trace=trace,
            sample_id=sample_id,
        )
        return per_metric_results

    def _determine_worker_count(self) -> int:
        if not self._instances:
            return 0
        if self._execution_controller is not None:
            lane_workers = self._execution_controller.metric_workers
            if lane_workers <= 1:
                return 1
            return max(1, min(lane_workers, len(self._instances)))
        if self._worker_hint is not None and self._worker_hint > 0:
            return max(1, min(self._worker_hint, len(self._instances)))
        cpu = os.cpu_count() or 1
        default = max(1, min(cpu, 8))
        return max(1, min(default, len(self._instances)))

    @staticmethod
    def _compose_cache_id(task_id: Optional[str], sample_id: str) -> str:
        if task_id:
            return f"{task_id}:{sample_id}"
        return sample_id

    @staticmethod
    def _compose_cache_namespace(task_id: Optional[str], judge_output: Dict) -> str:
        prefix = "judge" if judge_output else "task"
        suffix = task_id or "global"
        return f"{prefix}/{suffix}"

    def _resolve_cache_namespace(
        self,
        task_id: Optional[str],
        judge_output: Dict,
    ) -> str:
        runtime_context = self._aggregation_runtime_context
        if runtime_context is not None and runtime_context.details_namespace:
            return runtime_context.details_namespace
        return self._compose_cache_namespace(task_id, judge_output)

    @staticmethod
    def _resolve_sample_outputs(
        *,
        sample: dict,
        model_output: Dict,
        judge_output: Dict,
    ) -> Tuple[Dict, Dict, Dict | list]:
        """Normalize model/judge payloads before persistence and metric evaluation."""

        resolved_model_output = resolve_model_output(sample, model_output)
        resolved_judge_output = resolve_judge_output(sample, judge_output)
        resolved_arena_trace = resolve_arena_trace(sample, resolved_model_output)
        if resolved_arena_trace or "arena_trace" in resolved_model_output:
            normalized_model_output = dict(resolved_model_output)
            normalized_model_output["arena_trace"] = resolved_arena_trace
            resolved_model_output = normalized_model_output
        return resolved_model_output, resolved_judge_output, resolved_arena_trace


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
