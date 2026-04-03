"""Terminal benchmark metrics."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from gage_eval.metrics.base import MetricContext, SimpleMetric
from gage_eval.registry import registry


def _resolve_terminal_bench_flag(context: MetricContext, key: str) -> Any:
    value = context.get(f"judge_output.{key}")
    if value is not None:
        return value
    return context.get(f"judge_output.raw_output.{key}")


@registry.asset(
    "metrics",
    "terminal_bench_resolve_rate",
    desc="Terminal Bench Pass@1 (resolve rate)",
    tags=("terminal_bench",),
    default_aggregation="mean",
)
class TerminalBenchResolveRateMetric(SimpleMetric):
    """Resolve rate derived from terminal benchmark verifier output."""

    value_key = "resolve_rate"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        resolved = bool(_resolve_terminal_bench_flag(context, "resolved"))
        failure_reason = _resolve_terminal_bench_flag(context, "failure_reason")
        return (1.0 if resolved else 0.0), {
            "resolved": resolved,
            "failure_reason": failure_reason,
        }


@registry.asset(
    "metrics",
    "terminal_bench_failure_reason",
    desc="Terminal Bench failure reason distribution",
    tags=("terminal_bench", "analysis"),
    default_aggregation="categorical_count",
)
class TerminalBenchFailureReasonMetric(SimpleMetric):
    """Capture terminal benchmark failure_reason for categorical aggregation."""

    value_key = "count"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        resolved = bool(_resolve_terminal_bench_flag(context, "resolved"))
        reason = _resolve_terminal_bench_flag(context, "failure_reason")
        if resolved:
            return 0.0, {"failure_reason": None, "resolved": True}
        if not reason:
            reason = "unknown"
        return 1.0, {"failure_reason": str(reason), "resolved": False}


__all__ = [
    "TerminalBenchFailureReasonMetric",
    "TerminalBenchResolveRateMetric",
]
