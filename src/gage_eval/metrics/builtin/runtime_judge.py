"""Generic metrics derived from runtime-owned judge outputs."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from gage_eval.metrics.base import MetricContext, SimpleMetric


class RuntimeResolveRateMetric(SimpleMetric):
    """Resolve rate derived from `judge_output.resolved`."""

    value_key = "resolve_rate"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        """Compute the sample-level resolve indicator.

        Args:
            context: Metric evaluation context for the current sample.

        Returns:
            A numeric resolve flag plus diagnostic metadata.
        """

        resolved = bool(context.get("judge_output.resolved", False))
        failure_reason = context.get("judge_output.failure_reason")
        return (1.0 if resolved else 0.0), {
            "resolved": resolved,
            "failure_reason": failure_reason,
        }


class RuntimeFailureReasonMetric(SimpleMetric):
    """Failure reason distribution derived from runtime-owned judge output."""

    value_key = "count"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        """Compute the categorical failure bucket for one sample.

        Args:
            context: Metric evaluation context for the current sample.

        Returns:
            A count contribution plus the resolved failure category metadata.
        """

        resolved = bool(context.get("judge_output.resolved", False))
        reason = context.get("judge_output.failure_reason")
        if resolved:
            return 0.0, {"failure_reason": None, "resolved": True}
        if not reason:
            reason = "unknown"
        return 1.0, {"failure_reason": reason, "resolved": False}


__all__ = [
    "RuntimeResolveRateMetric",
    "RuntimeFailureReasonMetric",
]
