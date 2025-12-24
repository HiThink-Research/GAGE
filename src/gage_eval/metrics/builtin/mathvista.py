"""MathVista accuracy metric for mixed multiple-choice and open-ended QA."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.metrics.utils import ensure_list_of_strings, extract_field, normalize_text_advanced
from gage_eval.registry import registry


def _extract_choice_letter(prediction: str) -> Optional[str]:
    """Extract the last obvious multiple-choice letter from a model prediction."""

    patterns = [
        r"\\boxed\{\s*([A-Ea-e])\s*\}",
        r"<answer>\s*([A-Ea-e])",
        r"\b([A-Ea-e])\b",
    ]
    candidates: list[str] = []
    for pat in patterns:
        candidates.extend(re.findall(pat, prediction))
    # Flatten tuple matches returned by regex groups.
    flat: list[str] = []
    for cand in candidates:
        if isinstance(cand, tuple):
            flat.extend([c for c in cand if c])
        elif cand:
            flat.append(cand)
    return flat[-1] if flat else None


def _resolve_expected_label(
    answer: Any,
    option_map: Dict[str, Any],
    *,
    answer_index_base: int = 0,
) -> Optional[str]:
    if answer is None:
        return None
    # Direct letter answers (e.g., "A", "b").
    if isinstance(answer, str) and len(answer.strip()) == 1 and answer.strip().isalpha():
        return answer.strip().upper()
    # Index-based answers (some datasets use 0/1-based indices).
    if isinstance(answer, int):
        labels = list(option_map.keys())
        idx = answer - answer_index_base
        if 0 <= idx < len(labels):
            return str(labels[idx]).upper()
    # Text-match fallback: match normalized option content.
    answer_norm = normalize_text_advanced(str(answer), strip=True, collapse_whitespace=True)
    for label, text in option_map.items():
        if normalize_text_advanced(str(text), strip=True, collapse_whitespace=True) == answer_norm:
            return str(label).upper()
    return None


def _extract_numeric_answer(prediction: str, answer_type: str) -> Optional[str]:
    """Extract a numeric answer from free-form text (integer/float)."""
    if not prediction:
        return None

    # STEP 1: Try direct conversion (best effort).
    try:
        if answer_type == "integer":
            return str(int(float(prediction)))
        elif answer_type == "float":
            return str(float(prediction))
    except (ValueError, TypeError):
        pass

    # STEP 2: Extract the last numeric token with a simple regex.
    # Pattern: optional sign + digits + optional decimal part.
    # NOTE: This intentionally ignores more complex formats (e.g., thousand separators).
    numbers = re.findall(r"-?\d+(?:\.\d+)?", prediction)
    if not numbers:
        return None

    last_num = numbers[-1]
    try:
        if answer_type == "integer":
            return str(int(float(last_num)))
        elif answer_type == "float":
            return str(float(last_num))
    except (ValueError, TypeError):
        return None
    return None

@registry.asset(
    "metrics",
    "mathvista_accuracy",
    desc="MathVista mixed-type accuracy (MCQ letter-first, else normalized text match)",
    tags=("vision", "mathvista"),
    default_aggregation="mean",
)
class MathVistaAccuracyMetric(SimpleMetric):
    value_key = "acc"

    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: Resolve config fields.
        prediction_field = self.args.get("prediction_field", "model_output.answer")
        option_map_field = self.args.get("option_map_field", "sample.metadata.option_map")
        correct_choice_field = self.args.get("correct_choice_field", "sample.metadata.correct_choice")
        choices_field = self.args.get("choices_field", "sample.choices")
        answer_field = self.args.get("answer_field", "sample.answer")
        label_field = self.args.get("label_field", "sample.answer")
        # Optional: datasets may provide an explicit numeric answer type.
        answer_type_field = self.args.get("answer_type_field", "sample.answer_type")

        # STEP 2: Load inputs and normalize the prediction.
        prediction_raw = extract_field(context, prediction_field, default="")
        prediction = normalize_text_advanced(str(prediction_raw), strip=True, collapse_whitespace=True) or ""
        option_map = extract_field(context, option_map_field, default={}) or {}
        choices = extract_field(context, choices_field, default=[]) or []
        answer = extract_field(context, correct_choice_field)
        if answer is None:
            answer = extract_field(context, answer_field)
        if answer is None:
            answer = extract_field(context, "sample.label")
        if answer is None:
            answer = extract_field(context, "sample.metadata.answer")
        
        answer_type = extract_field(context, answer_type_field)

        is_multi_choice = bool(option_map) or bool(choices)

        if is_multi_choice:
            # STEP 3: Score multiple-choice questions via label matching.
            if not option_map and choices:
                option_map = {c.get("label"): extract_field(c, "message.content.0.text") for c in choices if c}
            option_map = {str(k).upper(): v for k, v in option_map.items() if k is not None}

            expected_label = _resolve_expected_label(answer, option_map, answer_index_base=0)
            pred_label = None
            if prediction and option_map:
                extracted = _extract_choice_letter(prediction)
                if extracted:
                    pred_label = extracted.upper()
                else:
                    # Fallback: match normalized option content.
                    pred_norm = normalize_text_advanced(prediction, strip=True, collapse_whitespace=True)
                    for label, text in option_map.items():
                        if normalize_text_advanced(str(text), strip=True, collapse_whitespace=True) == pred_norm:
                            pred_label = label.upper()
                            break
            matched = bool(expected_label and pred_label and expected_label == pred_label)
            score = 1.0 if matched else 0.0
            metadata = {
                "prediction_raw": prediction_raw,
                "prediction_label": pred_label,
                "expected_label": expected_label,
                "option_map": option_map,
            }
            return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)

        # STEP 4: For open-ended numeric questions, extract and compare numbers.
        if answer_type in ("integer", "float") and prediction:
            extracted_num = _extract_numeric_answer(prediction, answer_type)
            # Extract numeric reference as well.
            ref_raw = extract_field(context, label_field, default="")
            # NOTE: Even if the reference is a clean number, normalize it for safety.
            ref_str = str(ref_raw)
            ref_num = _extract_numeric_answer(ref_str, answer_type)
            
            if extracted_num is not None and ref_num is not None:
                # Numeric match: string equality is sufficient after normalization.
                matched = (extracted_num == ref_num)
                score = 1.0 if matched else 0.0
                return MetricResult(
                    sample_id=context.sample_id, 
                    values={self.value_key: score}, 
                    metadata={
                        "prediction": prediction, 
                        "extracted_prediction": extracted_num,
                        "reference": ref_num
                    }
                )

        # STEP 5: Fallback to normalized exact match for non-multiple-choice, non-numeric cases.
        references_raw = extract_field(context, label_field, default="")
        references = [
            normalize_text_advanced(text, strip=True, collapse_whitespace=True)
            for text in ensure_list_of_strings(references_raw)
        ]
        references = [r for r in references if r]
        matched = bool(prediction and references and any(prediction == ref for ref in references))
        score = 1.0 if matched else 0.0
        metadata = {"prediction": prediction, "references": references}
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)

# 2.benchmark MathVista
@registry.asset(
    "metrics",
    "mathvista_chat_accuracy",
    desc="MathVista chat accuracy (MCQ letter-first, else normalized text match)",
    tags=("vision", "mathvista"),
    default_aggregation="mean",
)
class MathVistaChataccuracyMetric(SimpleMetric):
    value_key = "acc"

    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: Resolve config fields.
        prediction_field = self.args.get("prediction_field", "model_output.answer")
        option_map_field = self.args.get("option_map_field", "sample.metadata.option_map")
        correct_choice_field = self.args.get("correct_choice_field", "sample.metadata.correct_choice")
        choices_field = self.args.get("choices_field", "sample.choices")
        answer_field = self.args.get("answer_field", "sample.answer")
        label_field = self.args.get("label_field", "sample.answer")
        # Optional: datasets may provide an explicit numeric answer type.
        answer_type_field = self.args.get("answer_type_field", "sample.answer_type")
        question_type_field  = self.args.get("question_type_field", "sample.question_type")
        shot_type_field  = self.args.get("shot_type_field", "sample.shot_type")
        # STEP 2: Load inputs and normalize prediction.
        answer_type = extract_field(context, answer_type_field)
        shot_type = extract_field(context, shot_type_field)
        question_type = extract_field(context, question_type_field)

        prediction_raw = extract_field(context, prediction_field, default="")

        if shot_type == 'code':
            prediction, error = evaluate_code(prediction_raw)
        else:
            prediction = normalize_text_advanced(str(prediction_raw), strip=True, collapse_whitespace=True) or ""
        option_map = extract_field(context, option_map_field, default={}) or {}
        choices = extract_field(context, choices_field, default=[]) or []
        answer = extract_field(context, correct_choice_field)
        if answer is None:
            answer = extract_field(context, answer_field)
        if answer is None:
            answer = extract_field(context, "sample.label")
        if answer is None:
            answer = extract_field(context, "sample.metadata.answer")


        is_multi_choice = question_type == 'multi_choice'

        if is_multi_choice:
            # STEP 3: Score multiple-choice questions via label matching.
            if not option_map and choices:
                option_map = {c.get("label"): extract_field(c, "message.content.0.text") for c in choices if c}
            option_map = {str(k).upper(): v for k, v in option_map.items() if k is not None}

            expected_label = _resolve_expected_label(answer, option_map, answer_index_base=0)
            pred_label = None
            if prediction and option_map:
                extracted = _extract_choice_letter(prediction)
                if extracted:
                    pred_label = extracted.upper()
                else:
                    # Fallback: match normalized option content.
                    pred_norm = normalize_text_advanced(prediction, strip=True, collapse_whitespace=True)
                    for label, text in option_map.items():
                        if normalize_text_advanced(str(text), strip=True, collapse_whitespace=True) == pred_norm:
                            pred_label = label.upper()
                            break
            matched = bool(expected_label and pred_label and expected_label == pred_label)
            score = 1.0 if matched else 0.0
            metadata = {
                "prediction_raw": prediction_raw,
                "prediction_label": pred_label,
                "expected_label": expected_label,
                "option_map": option_map,
            }
            return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)

        # STEP 4: For open-ended numeric questions, extract and compare numbers.
        if answer_type in ("integer", "float") and prediction:
            extracted_num = _extract_numeric_answer(prediction, answer_type)
            # Extract numeric reference as well.
            ref_raw = extract_field(context, label_field, default="")
            # NOTE: Even if the reference is a clean number, normalize it for safety.
            ref_str = str(ref_raw)
            ref_num = _extract_numeric_answer(ref_str, answer_type)
            
            if extracted_num is not None and ref_num is not None:
                # Numeric match: string equality is sufficient after normalization.
                matched = (extracted_num == ref_num)
                score = 1.0 if matched else 0.0
                return MetricResult(
                    sample_id=context.sample_id, 
                    values={self.value_key: score}, 
                    metadata={
                        "prediction": prediction, 
                        "extracted_prediction": extracted_num,
                        "reference": ref_num
                    }
                )

        # STEP 5: Fallback to normalized exact match for non-multiple-choice, non-numeric cases.
        references_raw = extract_field(context, label_field, default="")
        references = [
            normalize_text_advanced(text, strip=True, collapse_whitespace=True)
            for text in ensure_list_of_strings(references_raw)
        ]
        references = [r for r in references if r]
        matched = bool(prediction and references and any(prediction == ref for ref in references))
        score = 1.0 if matched else 0.0
        metadata = {"prediction": prediction, "references": references}
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)


__all__ = ["MathVistaAccuracyMetric", "MathVistaChataccuracyMetric"]
