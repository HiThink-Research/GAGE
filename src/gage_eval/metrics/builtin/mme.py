"""MME (Multimodal Evaluation) accuracy metric for Yes/No questions.

Reference: Awesome-Multimodal-Large-Language-Models/eval_tool/calculation.py
MME dataset: one image corresponds to two questions (same question_id).
Evaluation metrics:
- acc: per-question accuracy
- acc_plus: per-image accuracy (both questions correct)
"""

from __future__ import annotations

from typing import Any, Optional

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.metrics.utils import (
    extract_field,
    get_text_content_of_first_predict_result,
    get_first_reference,
)
from gage_eval.registry import registry
from loguru import logger


def _parse_pred_ans(pred_ans: str) -> str:
    """Parse prediction answer to yes/no/other (like calculation.py).
    
    Args:
        pred_ans: Raw prediction text.
        
    Returns:
        "yes", "no", or "other".
    """
    if not pred_ans:
        return "other"
    
    pred_ans_lower = pred_ans.lower().strip()
    
    # Direct matches
    if pred_ans_lower in ("yes", "no"):
        return pred_ans_lower
    
    # Check first 4 characters for "yes" or "no" (like calculation.py)
    prefix_pred_ans = pred_ans_lower[:4]
    
    if "yes" in prefix_pred_ans:
        return "yes"
    elif "no" in prefix_pred_ans:
        return "no"
    else:
        return "other"


def _normalize_yes_no(text: str) -> Optional[str]:
    """Normalize text to Yes/No (for reference normalization).
    
    Args:
        text: Input text to normalize.
        
    Returns:
        "Yes", "No", or None if cannot determine.
    """
    if not text:
        return None
    
    text_lower = text.strip().lower()
    
    # Direct matches
    if text_lower in ("yes", "y", "1", "true"):
        return "Yes"
    if text_lower in ("no", "n", "0", "false"):
        return "No"
    
    return None


@registry.asset(
    "metrics",
    "mme_accuracy",
    desc="MME accuracy metric for Yes/No questions (per-question and per-image accuracy)",
    tags=("vision", "mme"),
    default_aggregation="mean",
)
class MMEAccuracyMetric(SimpleMetric):
    """Accuracy metric for MME dataset (Yes/No questions).
    
    Like MathVistaChataccuracyMetric, extracts prediction and reference from sample.
    Supports per-question accuracy (acc) and per-image accuracy (acc_plus).
    """
    
    value_key = "acc"

    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: Extract sample/predict/ground truth
        sample_dict = extract_field(context, 'sample')
        references_raw = extract_field(sample_dict, "references") or []
        
        # MME: each sample contains one question
        # Get single reference answer
        ref_answer = get_first_reference(sample_dict) if references_raw else None
        if not ref_answer and isinstance(references_raw, list) and len(references_raw) > 0:
            ref_answer = references_raw[0]
        
        prediction_raw = get_text_content_of_first_predict_result(sample_dict)
        
        # Extract question_id for grouping samples by image (for acc_plus calculation)
        question_id = extract_field(sample_dict, "metadata.question_id") or None

        # STEP 2: Parse prediction from model output (single answer)
        pred_text = str(prediction_raw) if prediction_raw else ""
        pred_ans = _parse_pred_ans(pred_text)
        
        # STEP 3: Compare prediction with reference
        ref_norm = _normalize_yes_no(str(ref_answer)) if ref_answer else None
        ref_ans = ref_norm.lower() if ref_norm else None
        
        is_correct = False
        if ref_ans is None:
            grading_method = "invalid_reference"
        elif pred_ans == "other":
            grading_method = "other_prediction"
        else:
            is_correct = (pred_ans == ref_ans)
            grading_method = "yes_no_match"
        
        # STEP 4: Build metadata
        metadata = {
            "prediction": pred_text,
            "prediction_parsed": pred_ans,
            "reference": ref_answer,
            "reference_normalized": ref_ans,
            "is_correct": is_correct,
            "grading_method": grading_method,
        }
        if question_id:
            metadata["question_id"] = question_id

        # Return per-question accuracy (1.0 if correct, 0.0 otherwise)
        # acc_plus will be calculated in aggregator by grouping by question_id
        return MetricResult(
            sample_id=context.sample_id,
            values={self.value_key: 1.0 if is_correct else 0.0},
            metadata=metadata,
        )



__all__ = ["MMEAccuracyMetric"]
