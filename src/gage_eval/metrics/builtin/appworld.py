"""AppWorld metrics."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from gage_eval.metrics.base import MetricContext, SimpleMetric
from gage_eval.registry import registry


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_count(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return float(len(value))
    if isinstance(value, dict):
        return float(len(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@registry.asset(
    "metrics",
    "appworld_tgc",
    desc="AppWorld Task Goal Completion (TGC)",
    tags=("appworld",),
    default_aggregation="mean",
)
class AppWorldTGCMetric(SimpleMetric):
    value_key = "tgc"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        tgc = _coerce_float(context.get("judge_output.appworld.tgc"))
        if tgc is None:
            return 0.0, {"missing_tgc": True}
        return tgc, {"missing_tgc": False}


@registry.asset(
    "metrics",
    "appworld_sgc",
    desc="AppWorld Scenario Goal Completion (SGC)",
    tags=("appworld",),
    default_aggregation="mean",
)
class AppWorldSGCMetric(SimpleMetric):
    value_key = "sgc"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        sgc = _coerce_float(context.get("judge_output.appworld.sgc"))
        if sgc is None:
            return 0.0, {"missing_sgc": True}
        return sgc, {"missing_sgc": False}


@registry.asset(
    "metrics",
    "appworld_success_rate",
    desc="AppWorld success rate (tgc >= 1)",
    tags=("appworld",),
    default_aggregation="mean",
)
class AppWorldSuccessMetric(SimpleMetric):
    value_key = "success"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        tgc = _coerce_float(context.get("judge_output.appworld.tgc"))
        if tgc is None:
            return 0.0, {"missing_tgc": True}
        return (1.0 if tgc >= 1.0 else 0.0), {"missing_tgc": False, "tgc": tgc}


@registry.asset(
    "metrics",
    "appworld_pass_count",
    desc="AppWorld pass count (number of passed assertions)",
    tags=("appworld",),
    default_aggregation="mean",
)
class AppWorldPassCountMetric(SimpleMetric):
    value_key = "passes"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        passes = _coerce_count(context.get("judge_output.appworld.tests.passes"))
        if passes is None:
            return 0.0, {"missing_passes": True}
        return passes, {"missing_passes": False}


@registry.asset(
    "metrics",
    "appworld_fail_count",
    desc="AppWorld fail count (number of failed assertions)",
    tags=("appworld",),
    default_aggregation="mean",
)
class AppWorldFailCountMetric(SimpleMetric):
    value_key = "fails"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        fails = _coerce_count(context.get("judge_output.appworld.tests.fails"))
        if fails is None:
            return 0.0, {"missing_fails": True}
        return fails, {"missing_fails": False}


@registry.asset(
    "metrics",
    "appworld_difficulty",
    desc="AppWorld difficulty distribution",
    tags=("appworld",),
    default_aggregation="categorical_count",
)
class AppWorldDifficultyMetric(SimpleMetric):
    value_key = "count"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        difficulty = context.get("judge_output.appworld.difficulty")
        category_field = str(self.args.get("category_field", "difficulty"))
        if difficulty is None:
            return 0.0, {"missing_difficulty": True}
        return 1.0, {category_field: difficulty, "missing_difficulty": False}


__all__ = [
    "AppWorldTGCMetric",
    "AppWorldSGCMetric",
    "AppWorldSuccessMetric",
    "AppWorldPassCountMetric",
    "AppWorldFailCountMetric",
    "AppWorldDifficultyMetric",
]
