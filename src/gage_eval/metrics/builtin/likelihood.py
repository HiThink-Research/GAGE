"""Likelihood / Perplexity style metrics."""

from __future__ import annotations

import math
from typing import Any

from gage_eval.metrics.base import MetricContext, SimpleMetric
from gage_eval.metrics.utils import extract_field, flatten_numeric_list
from gage_eval.registry import registry


@registry.asset(
    "metrics",
    "likelihood",
    desc="Compute NLL/PPL from loss or token logprobs",
    tags=("likelihood", "ppl"),
    default_aggregation="mean",
)
class LikelihoodMetric(SimpleMetric):
    """Computes NLL/PPL from `loss` or `token_logprobs` fields."""

    value_key = "nll"

    def compute_value(self, context: MetricContext) -> tuple[float, dict]:
        metric_type = str(self.args.get("metric_type", "nll")).lower()
        loss_field = self.args.get("loss_field", "model_output.loss")
        logprob_field = self.args.get("logprob_field", "model_output.token_logprobs")

        loss_value = extract_field(context, loss_field)
        source = None
        metadata = {"metric_type": metric_type}

        if loss_value is not None:
            try:
                loss = float(loss_value)
                source = "loss"
            except (TypeError, ValueError):
                loss = None
        else:
            loss = None

        if loss is None:
            logprobs_raw = extract_field(context, logprob_field)
            logprobs = flatten_numeric_list(logprobs_raw)
            if logprobs:
                # NOTE: Use negative log-likelihood averaged over tokens.
                avg_logprob = sum(logprobs) / len(logprobs)
                loss = -avg_logprob
                source = "token_logprobs"
                metadata["token_count"] = len(logprobs)

        if loss is None:
            metadata["error"] = "missing_loss_or_logprobs"
            return 0.0, metadata

        metadata["source"] = source
        if metric_type == "ppl":
            ppl = math.exp(loss)
            return float(ppl), metadata
        return float(loss), metadata
