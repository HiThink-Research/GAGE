"""SimpleQA Verified accuracy metric for factuality evaluation."""

from __future__ import annotations

from typing import Any, Mapping

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.metrics.utils import (
    ensure_list_of_strings,
    extract_field,
    normalize_text_advanced,
    get_text_content_of_first_predict_result,
    get_first_reference,
)
from gage_eval.registry import registry

@registry.asset(
    "metrics",
    "simpleqa_verified_accuracy",
    desc="SimpleQA Verified accuracy metric for factuality evaluation",
    tags=("factuality", "simpleqa_verified", "question_answering"),
    default_aggregation="mean",
)
class SimpleQAVerifiedAccuracyMetric(SimpleMetric):
    """Metric for evaluating SimpleQA Verified answers using normalized text matching."""

    value_key = "acc"

    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: Resolve config fields.
        prediction_field = self.args.get("prediction_field", "model_output.answer")
        reference_field = self.args.get("reference_field", "sample.references")
        label_field = self.args.get("label_field", "sample.label")
        case_sensitive = self.args.get("case_sensitive", False)
        strip_whitespace = self.args.get("strip_whitespace", True)
        collapse_whitespace = self.args.get("collapse_whitespace", True)

        # STEP 2: Extract prediction and reference
        sample_dict = extract_field(context, "sample", default={})
        prediction_raw = extract_field(context, prediction_field, default="")
        if isinstance(sample_dict, dict) and "predict_result" in sample_dict:
            prediction_raw = get_text_content_of_first_predict_result(sample_dict) or prediction_raw
        
        prediction = str(prediction_raw) if prediction_raw else ""

        references_raw = (extract_field(context, reference_field, default=None) or
                         (get_first_reference(sample_dict) if isinstance(sample_dict, dict) else None) or
                         extract_field(context, label_field, default=""))
        references = ensure_list_of_strings(references_raw)
        
        if not references:
            return MetricResult(
                sample_id=context.sample_id,
                values={self.value_key: 0.0},
                metadata={"prediction": prediction, "reference": None},
            )

        # STEP 3: Normalize and compare
        pred_norm = normalize_text_advanced(
            prediction,
            case_sensitive=case_sensitive,
            strip=strip_whitespace,
            collapse_whitespace=collapse_whitespace,
        ) or ""
        
        refs_norm = [
            normalize_text_advanced(
                r,
                case_sensitive=case_sensitive,
                strip=strip_whitespace,
                collapse_whitespace=collapse_whitespace,
            ) or ""
            for r in references
        ]
        refs_norm = [r for r in refs_norm if r]
        
        # Check if prediction matches any reference
        is_correct = bool(pred_norm and refs_norm and any(pred_norm == r for r in refs_norm))

        return MetricResult(
            sample_id=context.sample_id,
            values={self.value_key: 1.0 if is_correct else 0.0},
            metadata={
                "prediction": prediction,
                "prediction_normalized": pred_norm,
                "reference": references[0] if references else None,
                "reference_normalized": refs_norm[0] if refs_norm else None,
            },
        )


@registry.asset(
    "metrics",
    "simpleqa_verified_judge_accuracy",
    desc="SimpleQA Verified accuracy metric using judge model verdicts",
    tags=("factuality", "simpleqa_verified", "question_answering", "judge"),
    default_aggregation="mean",
)
class SimpleQAVerifiedJudgeAccuracyMetric(SimpleMetric):
    """Metric for evaluating SimpleQA Verified answers using judge model verdicts."""

    value_key = "acc"

    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: Extract judge_output verdict.
        judge_output: Mapping[str, Any] = context.judge_output or {}
        verdict_raw = None
        
        if isinstance(judge_output, Mapping):
            # First try to get verdict directly from judge_output dict
            verdict_raw = judge_output.get("verdict") or judge_output.get("label")
            
            # If not found, try parsing the "answer" field as JSON (for JSON response_format)
            if verdict_raw is None:
                answer = judge_output.get("answer", "")
                if answer and isinstance(answer, str):
                    answer_str = answer.strip()
                    # Try to parse as JSON
                    try:
                        import json
                        parsed = json.loads(answer_str)
                        if isinstance(parsed, dict):
                            verdict_raw = parsed.get("verdict") or parsed.get("label")
                    except (json.JSONDecodeError, ValueError, TypeError):
                        # Kaggle/original grader returns A/B/C only.
                        if answer_str in ("A", "B", "C"):
                            verdict_raw = answer_str
                        else:
                            # Otherwise treat the entire answer as verdict (fallback)
                            verdict_raw = answer_str
        
        verdict = str(verdict_raw or "").strip().upper()
        # Map Kaggle/original grader letter outputs to verdict strings.
        if verdict in ("A", "B", "C"):
            verdict = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}[verdict]

        # STEP 2: Map verdict to correctness.
        # CORRECT -> 1.0, INCORRECT / NOT_ATTEMPTED / other -> 0.0
        is_correct = verdict == "CORRECT"

        return MetricResult(
            sample_id=context.sample_id,
            values={self.value_key: 1.0 if is_correct else 0.0},
            metadata={
                "verdict": verdict or None,
            },
        )


__all__ = ["SimpleQAVerifiedAccuracyMetric", "SimpleQAVerifiedJudgeAccuracyMetric"]
