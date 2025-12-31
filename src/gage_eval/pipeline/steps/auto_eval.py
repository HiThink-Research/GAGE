"""Auto evaluation step that records per-sample metric values."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.observability.decorators import observable_stage
from gage_eval.observability.logger import ObservableLogger
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.evaluation.cache import EvalCache
from gage_eval.metrics import MetricRegistry, MetricInstance, MetricContext
from gage_eval.evaluation.sample_envelope import resolve_judge_output, resolve_model_output, snapshot_sample
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
    ) -> None:
        super().__init__("AutoEvalStep")
        self._registry = metric_registry or MetricRegistry()
        self._instances: List[MetricInstance] = [self._registry.build_metric(spec) for spec in metric_specs]
        self._cache = cache_store
        self._aggregate_lock = threading.Lock()
        self._logger = ObservableLogger()
        self._worker_hint = _env_int("GAGE_EVAL_AUTOEVAL_WORKERS")

    def has_metrics(self) -> bool:
        return bool(self._instances)

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
        resolved_model_output = resolve_model_output(sample, model_output)
        resolved_judge_output = resolve_judge_output(sample, judge_output)
        per_metric_results: Dict[str, Dict] = {}
        logger = self._logger
        worker_count = self._determine_worker_count()
        if self._instances:
            if worker_count <= 1 or len(self._instances) == 1:
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
                        worker_count,
                        sample_id,
                        sample,
                        resolved_model_output,
                        resolved_judge_output,
                        trace,
                    )
                )
        record = {
            "task_id": task_id,
            "sample": snapshot_sample(sample),
            "model_output": resolved_model_output,
            "judge_output": resolved_judge_output,
            "metrics": per_metric_results,
        }
        cache_id = self._compose_cache_id(task_id, sample_id)
        if self._cache:
            namespace = self._compose_cache_namespace(task_id, resolved_judge_output)
            self._cache.write_sample(cache_id, record, namespace=namespace)
            logger.debug(
                "auto_eval",
                "Cached auto-eval sample sample_id={} task_id={}",
                sample_id,
                task_id,
                trace=trace,
                sample_id=sample_id,
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
        worker_count: int,
        sample_id: str,
        sample: dict,
        model_output: Dict,
        judge_output: Dict,
        trace: ObservabilityTrace,
    ) -> Dict[str, Dict]:
        per_metric_results: Dict[str, Dict] = {}
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    self._evaluate_metric,
                    instance,
                    sample_id,
                    sample,
                    model_output,
                    judge_output,
                    trace,
                ): instance.spec.metric_id
                for instance in self._instances
            }
            for future in as_completed(future_map):
                metric_id, result_dict = future.result()
                per_metric_results[metric_id] = result_dict
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


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
