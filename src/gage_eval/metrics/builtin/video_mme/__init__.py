"""Video-MME accuracy metric."""

from __future__ import annotations

import re
from typing import Optional

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.metrics.utils import (
    extract_field,
    get_first_reference,
    get_text_content_of_first_predict_result,
    normalize_text_advanced,
)
from gage_eval.registry import registry


def _extract_answer_letter(prediction: str) -> Optional[str]:
    """Extract a single choice letter (A/B/C/D) from the model prediction.

    The logic is adapted from the official Video-MME evaluation script:
    https://github.com/thanku-all/parse_answer/blob/main/eval_your_results.py
    """
    s = prediction.strip()
    answer_prefixes = [
        "the best answer is",
        "the correct answer is",
        "the answer is",
        "the answer",
        "the best option is",
        "the correct option is",
        "best answer:",
        "best option:",
        "answer:",
        "option:",
        "the correct answer",
        "the correct option",
    ]
    for prefix in answer_prefixes:
        s = s.replace(prefix, "")
    s = s.strip()

    # If the text is long and contains no letter, treat it as unparseable.
    if len(s.split()) > 10 and not re.search(r"[ABCD]", s, re.IGNORECASE):
        return ""

    matches = re.search(r"[ABCD]", s, re.IGNORECASE)
    if matches is None:
        return ""
    return matches.group(0).upper()


@registry.asset(
    "metrics",
    "video_mme_accuracy",
    desc="Video-MME accuracy (extract A/B/C/D and compare)",
    tags=("video", "video_mme"),
    default_aggregation="mean",
)
class VideoMMEAccuracyMetric(SimpleMetric):
    """Metric for evaluating Video-MME answers.

    Video-MME is a multiple-choice video understanding benchmark.  The ground
    truth is a single capital letter (A/B/C/D).  This metric extracts the first
    occurrence of A/B/C/D from the model response and compares it with the
    reference letter.
    """

    value_key = "acc"

    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: Extract prediction and reference.
        sample_dict = extract_field(context, "sample")
        answer = get_first_reference(sample_dict)
        prediction_raw = get_text_content_of_first_predict_result(sample_dict)

        prediction = normalize_text_advanced(
            str(prediction_raw or ""), strip=True, collapse_whitespace=True
        )

        # STEP 2: Extract choice letter.
        pred_label = _extract_answer_letter(prediction)
        ref_label = str(answer).strip().upper() if answer else ""

        # STEP 3: Score.
        matched = bool(
            pred_label and ref_label and pred_label.upper() == ref_label
        )
        score = 1.0 if matched else 0.0

        metadata = {
            "prediction_raw": prediction_raw,
            "prediction_label": pred_label,
            "expected_label": ref_label,
        }
        return MetricResult(
            sample_id=context.sample_id,
            values={self.value_key: score},
            metadata=metadata,
        )


__all__ = ["VideoMMEAccuracyMetric"]
