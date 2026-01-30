"""Tau2-specific aggregators."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from loguru import logger

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.aggregators import MetricAggregator
from gage_eval.metrics.base import AggregatedMetric, MetricResult


class Tau2PassHatAggregator(MetricAggregator):
    """Compute pass-hat@k (pass^k) for Tau2 tasks.

    This aggregation groups samples by task_id and estimates pass@k using
    the binomial estimator described in the Tau2 evaluation protocol.
    """

    def __init__(self, spec: MetricSpec) -> None:
        super().__init__(spec)
        self._results: List[MetricResult] = []
        self._task_field = str(spec.params.get("task_field", "task_id"))
        self._value_key = str(spec.params.get("value_key", "pass"))
        self._max_k = _coerce_int(spec.params.get("max_k"))
        self._k_values = _parse_k_values(spec.params.get("k_values"))

    def add(self, result: MetricResult) -> None:
        self._results.append(result)

    def finalize(self) -> AggregatedMetric:
        # STEP 1: Group per-sample pass indicators by task_id.
        task_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
        skipped_missing_reward = 0
        missing_task_id = 0
        total_samples = 0

        for result in self._results:
            total_samples += 1
            if result.metadata.get("missing_reward"):
                skipped_missing_reward += 1
                continue
            task_id = result.metadata.get(self._task_field)
            if not task_id:
                missing_task_id += 1
                continue
            stats = task_stats[str(task_id)]
            stats["total"] += 1
            if _is_success_value(result.values.get(self._value_key, 0.0)):
                stats["success"] += 1

        # STEP 2: Resolve k-values and compute pass-hat@k.
        min_trials = _resolve_min_trials(task_stats)
        k_values = _resolve_k_values(min_trials, self._max_k, self._k_values)
        values = {f"pass_hat@{k}": _pass_hat_k(task_stats, k) for k in k_values}

        used_samples = sum(stats["total"] for stats in task_stats.values())
        logger.debug(
            "Tau2PassHatAggregator finalized metric={} tasks={} samples={} used_samples={} min_trials={} k_values={}",
            self.spec.metric_id,
            len(task_stats),
            total_samples,
            used_samples,
            min_trials,
            k_values,
        )

        return AggregatedMetric(
            metric_id=self.spec.metric_id,
            aggregation=self.spec.aggregation or "tau2_pass_hat",
            values=values,
            count=used_samples,
            metadata={
                "total_samples": total_samples,
                "used_samples": used_samples,
                "missing_task_id": missing_task_id,
                "skipped_missing_reward": skipped_missing_reward,
                "task_count": len(task_stats),
                "min_trials": min_trials,
                "k_values": k_values,
            },
        )


def _resolve_min_trials(task_stats: Dict[str, Dict[str, int]]) -> int:
    if not task_stats:
        return 0
    return min(stats["total"] for stats in task_stats.values())


def _resolve_k_values(
    min_trials: int, max_k: Optional[int], k_values: Optional[Iterable[int]]
) -> List[int]:
    if min_trials <= 0:
        return []
    if k_values:
        filtered = [k for k in k_values if 1 <= k <= min_trials]
        return sorted(set(filtered))
    if max_k is not None:
        return list(range(1, min(min_trials, max_k) + 1))
    return list(range(1, min_trials + 1))


def _pass_hat_k(task_stats: Dict[str, Dict[str, int]], k: int) -> float:
    if k <= 0:
        return 0.0
    values: List[float] = []
    for stats in task_stats.values():
        total = stats["total"]
        success = stats["success"]
        if total < k:
            continue
        if success < k:
            values.append(0.0)
        else:
            values.append(math.comb(success, k) / math.comb(total, k))
    if not values:
        return 0.0
    return sum(values) / len(values)


def _is_success_value(value: float) -> bool:
    return (1.0 - 1e-6) <= float(value) <= (1.0 + 1e-6)


def _coerce_int(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_k_values(value: Optional[object]) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return [int(item) for item in items]
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return None


__all__ = ["Tau2PassHatAggregator"]
