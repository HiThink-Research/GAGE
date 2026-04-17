"""AIME accuracy metrics."""

from __future__ import annotations

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.metrics.filter.base import RegexFilter
from gage_eval.metrics.numeric import extract_numeric_answer
from gage_eval.metrics.match import match_str
from gage_eval.metrics.utils import (
    extract_field,
    get_text_content_of_first_predict_result,
    get_first_reference,
)
from gage_eval.registry import registry


@registry.asset(
    "metrics",
    "aime2024_accuracy",
    desc="AIME2024 accuracy",
    tags=("aime2024",),
    default_aggregation="mean",
)
class AIME2024AccuracyMetric(SimpleMetric):
    value_key = "acc"
    regex_pattern = r"ANSWER[：:]\s*(.*)"

    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: extract sample/predict /ground truth
        sample_dict = extract_field(context, "sample")
        answer = get_first_reference(sample_dict)
        prediction_raw = get_text_content_of_first_predict_result(sample_dict)

        # STEP 2: pretty prediction
        rf = RegexFilter(
            regex_pattern=self.regex_pattern, group_select=-1, ignore_case=True
        )
        pred = rf.apply(prediction_raw)

        pred = extract_numeric_answer(pred)

        # STEP 3: compute score
        final_pred, score = match_str(
            pred, str(answer), location="exact", numeric=True
        )
        score = float(score)
        metadata = {"prediction": final_pred, "references": answer}
        return MetricResult(
            sample_id=context.sample_id,
            values={self.value_key: score},
            metadata=metadata,
        )


@registry.asset(
    "metrics",
    "aime2026_accuracy",
    desc="AIME2026 accuracy",
    tags=("aime2026",),
    default_aggregation="mean",
)
class AIME2026AccuracyMetric(SimpleMetric):
    value_key = "acc"
    regex_pattern = r"ANSWER[：:]\s*(.*)"

    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: extract sample/predict /ground truth
        sample_dict = extract_field(context, "sample")
        answer = get_first_reference(sample_dict)
        prediction_raw = get_text_content_of_first_predict_result(sample_dict)

        # STEP 2: pretty prediction
        rf = RegexFilter(
            regex_pattern=self.regex_pattern, group_select=-1, ignore_case=True
        )
        pred = rf.apply(prediction_raw)

        pred = extract_numeric_answer(pred)

        # STEP 3: compute score
        final_pred, score = match_str(
            pred, str(answer), location="exact", numeric=True
        )
        score = float(score)
        metadata = {"prediction": final_pred, "references": answer}
        return MetricResult(
            sample_id=context.sample_id,
            values={self.value_key: score},
            metadata=metadata,
        )


__all__ = [
    "AIME2024AccuracyMetric",
    "AIME2026AccuracyMetric",
]
