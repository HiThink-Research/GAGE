"""SkillsBench metrics."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from gage_eval.metrics.base import MetricContext, SimpleMetric
from gage_eval.registry import registry


def _resolve_skillsbench_flag(context: MetricContext, key: str) -> Any:
    value = context.get(f"judge_output.{key}")
    if value is not None:
        return value
    return context.get(f"judge_output.raw_output.{key}")


@registry.asset(
    "metrics",
    "skillsbench_resolve_rate",
    desc="SkillsBench Pass@1 (resolve rate)",
    tags=("skillsbench",),
    default_aggregation="mean",
)
class SkillsBenchResolveRateMetric(SimpleMetric):
    """Resolve rate derived from SkillsBench verifier output."""

    value_key = "resolve_rate"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        resolved = bool(_resolve_skillsbench_flag(context, "resolved"))
        failure_reason = _resolve_skillsbench_flag(context, "failure_reason")
        return (1.0 if resolved else 0.0), {
            "resolved": resolved,
            "failure_reason": failure_reason,
        }


@registry.asset(
    "metrics",
    "skillsbench_failure_reason",
    desc="SkillsBench failure reason distribution",
    tags=("skillsbench", "analysis"),
    default_aggregation="categorical_count",
)
class SkillsBenchFailureReasonMetric(SimpleMetric):
    """Capture failure_reason for categorical aggregation."""

    value_key = "count"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        resolved = bool(_resolve_skillsbench_flag(context, "resolved"))
        reason = _resolve_skillsbench_flag(context, "failure_reason")
        if resolved:
            return 0.0, {"failure_reason": None, "resolved": True}
        if not reason:
            reason = "unknown"
        return 1.0, {"failure_reason": str(reason), "resolved": False}


__all__ = [
    "SkillsBenchFailureReasonMetric",
    "SkillsBenchResolveRateMetric",
]
