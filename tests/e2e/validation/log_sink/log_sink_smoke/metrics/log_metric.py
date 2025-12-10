"""Metric that logs via ObservableLogger to validate the log sink."""

from __future__ import annotations

from loguru import logger

from gage_eval.metrics.base import MetricContext, SimpleMetric
from gage_eval.registry import registry


@registry.asset(
    "metrics",
    "log_sink_metric",
    desc="Emits loguru events during metric computation。",
    tags=("test",),
    default_aggregation="mean",
)
class LogSinkMetric(SimpleMetric):
    """Metric that logs a message and compares model output with label."""

    value_key = "acc"

    def compute_value(self, context: MetricContext) -> float:
        sample_id = context.sample_id
        logger.bind(stage="log_sink_metric", sample_id=sample_id).info(
            "Log sink smoke metric executed for sample_id={}", sample_id
        )
        prediction = str(context.model_output.get("answer", "")).strip()
        label = str(context.sample.get("label", "")).strip()
        return 1.0 if prediction == label else 0.0
