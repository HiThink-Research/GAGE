"""SWE-bench Pro metrics."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from gage_eval.metrics.base import MetricContext, SimpleMetric
from gage_eval.registry import registry


@registry.asset(
    "metrics",
    "swebench_resolve_rate",
    desc="SWE-bench Pro Pass@1 (resolve rate)",
    tags=("swebench",),
    default_aggregation="mean",
)
class SwebenchResolveRateMetric(SimpleMetric):
    """Resolve rate derived from judge_output.resolved."""

    value_key = "resolve_rate"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        resolved = bool(context.get("judge_output.resolved", False))
        failure_reason = context.get("judge_output.failure_reason")
        return (1.0 if resolved else 0.0), {
            "resolved": resolved,
            "failure_reason": failure_reason,
        }


@registry.asset(
    "metrics",
    "swebench_failure_reason",
    desc="SWE-bench Pro failure reason distribution",
    tags=("swebench", "analysis"),
    default_aggregation="categorical_count",
)
class SwebenchFailureReasonMetric(SimpleMetric):
    """Capture failure_reason for categorical aggregation."""

    value_key = "count"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        resolved = bool(context.get("judge_output.resolved", False))
        reason = context.get("judge_output.failure_reason")
        if resolved:
            return 0.0, {"failure_reason": None, "resolved": True}
        if not reason:
            reason = "unknown"
        return 1.0, {"failure_reason": reason, "resolved": False}


__all__ = [
    "SwebenchResolveRateMetric",
    "SwebenchFailureReasonMetric",
]
