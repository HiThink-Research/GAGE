"""GSM8K accuracy metric."""

from __future__ import annotations

import re
from typing import Optional

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


def _extract_boxed_answer(text: str) -> str:
    """Extract answer from LaTeX \\boxed{...} pattern.

    Handles nested braces correctly for expressions like \\boxed{\\frac{3}{4}}.
    Falls back to the original text if no \\boxed{} is found.

    Args:
        text: The text to extract answer from.

    Returns:
        The extracted answer or the original text if not found.
    """
    if not text:
        return text

    pattern = r"\\boxed\s*\{"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))

    if not matches:
        return text

    extracted_contents = []
    for match in matches:
        start_idx = match.end()
        brace_count = 1
        end_idx = start_idx

        while end_idx < len(text) and brace_count > 0:
            if text[end_idx] == "{":
                brace_count += 1
            elif text[end_idx] == "}":
                brace_count -= 1
            end_idx += 1

        if brace_count == 0:
            content = text[start_idx:end_idx - 1].strip()
            extracted_contents.append(content)

    if extracted_contents:
        return extracted_contents[-1]
    return text


@registry.asset(
    "metrics",
    "gsm8k_accuracy",
    desc="GSM8K accuracy",
    tags=("gsm8k", "math"),
    default_aggregation="mean",
)
class GSM8KAccuracyMetric(SimpleMetric):
    value_key = "acc"
    regex_pattern = r"ANSWER[：:]\s*(.*)"

    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: extract sample/prediction/ground truth
        sample_dict = extract_field(context, "sample")
        answer = get_first_reference(sample_dict)
        prediction_raw = get_text_content_of_first_predict_result(sample_dict)

        # STEP 2: try ANSWER: pattern first, then fallback to \boxed{}
        rf = RegexFilter(
            regex_pattern=self.regex_pattern, group_select=-1, ignore_case=True
        )
        pred = rf.apply(prediction_raw)

        if pred == prediction_raw:
            pred = _extract_boxed_answer(prediction_raw)

        pred = extract_numeric_answer(pred)

        # STEP 3: compute score
        answer_clean = str(answer).replace(",", "") if answer is not None else ""
        final_pred, score = match_str(
            pred, answer_clean, location="exact", numeric=True
        )
        score = float(score)
        metadata = {"prediction": final_pred, "references": answer_clean}
        return MetricResult(
            sample_id=context.sample_id,
            values={self.value_key: score},
            metadata=metadata,
        )


__all__ = ["GSM8KAccuracyMetric"]
