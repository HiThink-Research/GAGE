"""Built-in metrics for text and generic scalar comparisons."""

from __future__ import annotations

import re
from typing import Any, Optional

from gage_eval.metrics.base import (
    MetricContext,
    ComparisonMetric,
    SimpleMetric,
    NumericThresholdMetric,
)
from gage_eval.metrics.utils import extract_field, normalize_text_advanced
from gage_eval.registry import registry


@registry.asset(
    "metrics",
    "exact_match",
    desc="Strict text match metric",
    tags=("text",),
    default_aggregation="mean",
)
class ExactMatchMetric(ComparisonMetric):
    """Performs strict text matching between prediction and reference."""

    default_reference_field = "sample.label"
    default_prediction_field = "model_output.answer"

    def compare(self, prediction: Any, reference: Any) -> tuple[float, dict]:
        case_sensitive = bool(self.args.get("case_sensitive", False))
        strip = bool(self.args.get("strip_whitespace", True))
        normalized_ref = normalize_text_advanced(
            reference,
            case_sensitive=case_sensitive,
            strip=strip,
            collapse_whitespace=False,
        )
        normalized_pred = normalize_text_advanced(
            prediction,
            case_sensitive=case_sensitive,
            strip=strip,
            collapse_whitespace=False,
        )
        if normalized_ref is None or normalized_pred is None:
            return 0.0, {}
        return (1.0 if normalized_ref == normalized_pred else 0.0), {}


@registry.asset(
    "metrics",
    "contains",
    desc="Prediction contains reference text",
    tags=("text",),
    default_aggregation="mean",
)
class ContainsMatchMetric(ComparisonMetric):
    """Checks whether the prediction contains the reference text."""

    default_reference_field = "label"
    default_prediction_field = "model_output.answer"

    def compare(self, prediction: Any, reference: Any) -> tuple[float, dict]:
        case_sensitive = bool(self.args.get("case_sensitive", False))
        strip = bool(self.args.get("strip_whitespace", True))
        ref_norm = normalize_text_advanced(reference, case_sensitive=case_sensitive, strip=strip, collapse_whitespace=True) or ""
        pred_norm = normalize_text_advanced(prediction, case_sensitive=case_sensitive, strip=strip, collapse_whitespace=True) or ""
        score = 1.0 if ref_norm and ref_norm in pred_norm else 0.0
        return score, {}


@registry.asset(
    "metrics",
    "numeric_match",
    desc="Numeric match with tolerance",
    tags=("numeric",),
    default_aggregation="mean",
)
class NumericMatchMetric(NumericThresholdMetric):
    """Matches numeric answers with configurable tolerance."""

    default_reference_field = "label"
    default_prediction_field = "model_output.answer"


@registry.asset(
    "metrics",
    "regex_match",
    desc="Regex match metric",
    tags=("text", "regex"),
    default_aggregation="mean",
)
class RegexMatchMetric(SimpleMetric):
    """Matches using a regex pattern against a selected target field."""

    def setup(self) -> None:
        super().setup()
        pattern = self.args.get("pattern")
        if not pattern:
            raise ValueError("RegexMatchMetric requires 'pattern' in args")
        flags = 0
        if self.args.get("ignore_case", True):
            flags |= re.IGNORECASE
        self._compiled = re.compile(pattern, flags=flags)

    def compute_value(self, context: MetricContext) -> tuple[float, dict]:
        target_field = self.args.get("target_field", "model_output.answer")
        target = extract_field(context, target_field) or ""
        matched = bool(self._compiled.search(str(target)))
        return float(matched), {"target": target}


@registry.asset(
    "metrics",
    "judge_threshold",
    desc="Pass rate based on judge score threshold",
    tags=("judge",),
    default_aggregation="mean",
)
class JudgeThresholdMetric(NumericThresholdMetric):
    """Converts a judge score into a 0/1 decision using a threshold."""

    default_prediction_field = "judge_output.score"

    def extract_reference(self, context: MetricContext) -> Any:
        # NOTE: The threshold comes from metric args, not from the context fields.
        return self.args.get("threshold")

    def _get_threshold_config(self) -> dict:
        base = super()._get_threshold_config()
        base.update(
            {
                "mode": "ge",
                "threshold": self.args.get("threshold", 0.5),
            }
        )
        return base

    def compare(self, prediction: Any, reference: Any) -> tuple[float, dict]:
        score, metadata = super().compare(prediction, reference)
        metadata.setdefault("judge", prediction)
        return score, metadata


@registry.asset(
    "metrics",
    "text_length",
    desc="Text length metric (chars or words)",
    tags=("text", "analysis"),
    default_aggregation="mean",
)
class TextLengthMetric(SimpleMetric):
    """Computes text length using character or word units."""

    value_key = "length"

    def compute_value(self, context: MetricContext) -> tuple[float, dict]:
        target_field = self.args.get("target_field", "model_output.answer")
        unit = self.args.get("unit", "char")
        text_raw = extract_field(context, target_field) or ""
        text = str(text_raw)
        tokens = text.split() if unit == "word" else list(text)
        length = float(len(tokens))
        return length, {"unit": unit, "target": text}


@registry.asset(
    "metrics",
    "latency",
    desc="Extract latency from model output",
    tags=("latency",),
    default_aggregation="mean",
)
class LatencyMetric(SimpleMetric):
    """Extracts latency information from model/judge outputs."""

    value_key = "latency_ms"

    def compute_value(self, context: MetricContext) -> tuple[float, dict]:
        field = self.args.get("target_field", "model_output.latency_ms")
        default = float(self.args.get("default_latency", 0.0))
        value = extract_field(context, field, default=default)
        try:
            latency = float(value)
        except (TypeError, ValueError):
            latency = default
        return latency, {"target_field": field}
