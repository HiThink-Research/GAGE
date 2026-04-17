"""BeyondAIME accuracy metric with exact match grading."""

from __future__ import annotations

import re
from typing import Any, Optional

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.metrics.utils import (
    extract_field,
    get_text_content_of_first_predict_result,
    get_first_reference,
)
from gage_eval.registry import registry


def _extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from LaTeX \\boxed{...} pattern.
    
    Handles nested braces correctly for expressions like \\boxed{\\frac{3}{4}}.
    
    Args:
        text: The text to extract answer from.
        
    Returns:
        The extracted answer or None if not found.
    """
    if not text:
        return None
    
    # Find all occurrences of \boxed{ and extract with proper brace balancing
    pattern = r"\\boxed\s*\{"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    
    if not matches:
        return None
    
    # Process each match and extract content with balanced braces
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
            # Successfully found matching braces
            content = text[start_idx:end_idx - 1].strip()
            extracted_contents.append(content)
    
    if extracted_contents:
        # Return the last match (most likely the final answer)
        return extracted_contents[-1]
    return None


def _normalize_exact_match(s: str) -> str:
    """Normalize string for exact match comparison.
    
    For BeyondAIME, all answers are positive integers, so we:
    - Convert to lowercase
    - Remove all non-digit characters except minus sign
    - Keep only digits
    
    Args:
        s: The string to normalize.
        
    Returns:
        Normalized string containing only digits.
    """
    normalized = s.lower()
    # Remove letters and keep only digits
    normalized = re.sub(r"[a-z]", "", normalized)
    normalized = re.sub(r"[^0-9]+", "", normalized)
    return normalized


@registry.asset(
    "metrics",
    "beyond_aime_accuracy",
    desc="BeyondAIME accuracy with exact match grading",
    tags=("beyond_aime", "math"),
    default_aggregation="mean",
)
class BeyondAIMEAccuracyMetric(SimpleMetric):
    """Metric for evaluating BeyondAIME answers using exact match grading logic.
    
    BeyondAIME is a curated test set designed to benchmark advanced mathematical
    reasoning. Each problem's answer is a positive integer, allowing for an
    unambiguous and 100% accurate automated verification.
    """

    value_key = "acc"

    def compute(self, context: MetricContext) -> MetricResult:
        """Compute BeyondAIME accuracy metric.
        
        Args:
            context: The metric context containing sample and prediction data.
            
        Returns:
            MetricResult with accuracy score and metadata.
        """
        # STEP 1: Extract sample/prediction/ground truth
        sample_dict = extract_field(context, "sample")
        answer = get_first_reference(sample_dict)
        prediction_raw = get_text_content_of_first_predict_result(sample_dict)

        # STEP 2: Extract answer from prediction using boxed pattern
        pred = _extract_boxed_answer(prediction_raw) if prediction_raw else None
        
        # If no boxed answer found, use the raw prediction
        if pred is None:
            pred = prediction_raw.strip() if prediction_raw else ""

        # STEP 3: Compute score using exact match
        normalized_pred = _normalize_exact_match(str(pred))
        normalized_answer = _normalize_exact_match(str(answer))
        is_correct = normalized_pred == normalized_answer and normalized_pred != ""

        score = 1.0 if is_correct else 0.0
        metadata = {
            "prediction": pred,
            "references": answer,
            "normalized_prediction": normalized_pred if pred else None,
            "normalized_reference": normalized_answer,
        }
        
        return MetricResult(
            sample_id=context.sample_id,
            values={self.value_key: score},
            metadata=metadata,
        )


__all__ = ["BeyondAIMEAccuracyMetric"]
