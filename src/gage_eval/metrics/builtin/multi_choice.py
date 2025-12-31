"""Multiple-choice accuracy metric aligned with llm-eval logic."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence, Tuple

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.metrics.utils import (
    extract_field, normalize_text_advanced,
    get_text_content_of_first_predict_result,
    get_sample_label,
    get_first_reference,
)
from gage_eval.registry import registry

_BOXED_PATTERN = re.compile(r"\\boxed\s*\{([^}]*)\}", re.IGNORECASE)
_TOKEN_SPLIT = re.compile(r"[^A-Z]+")


def _normalize_option_map(raw: Any) -> Mapping[str, str]:
    if isinstance(raw, Mapping):
        return {str(key).upper(): str(value) for key, value in raw.items()}
    if isinstance(raw, Sequence):
        normalized = {}
        for entry in raw:
            if isinstance(entry, Mapping):
                label = entry.get("label") or entry.get("index")
                text = entry.get("text") or entry.get("value") or entry.get("content")
                if label is not None and text is not None:
                    normalized[str(label).upper()] = str(text)
        if normalized:
            return normalized
    return {}


def _normalize_choice_label(value: Any, allowed_letters: Sequence[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    direct = text.strip(" .:：()[]{}<>\"'")
    if direct in allowed_letters:
        return direct
    for prefix in ("OPTION", "CHOICE", "ANSWER", "答案", "选项"):
        if direct.startswith(prefix):
            tail = direct[len(prefix) :].strip(" .:：()[]{}<>\"'")
            if tail in allowed_letters:
                return tail
    for token in _TOKEN_SPLIT.split(text):
        if token in allowed_letters:
            return token
    return None


def _match_option_text(value: Any, option_map: Mapping[str, str]) -> Optional[str]:
    if value is None:
        return None
    normalized_value = normalize_text_advanced(value, collapse_whitespace=True) or ""
    for label, text in option_map.items():
        option_norm = normalize_text_advanced(text, collapse_whitespace=True) or ""
        if not option_norm:
            continue
        if normalized_value == option_norm or option_norm in normalized_value:
            return label
    return None

def _extract_prediction_letter(raw_answer: Any, allowed_letters: Sequence[str]) -> Optional[str]:
    if raw_answer is None:
        return None
    text = str(raw_answer).strip()
    candidates = []
    matches = _BOXED_PATTERN.findall(text)
    if matches:
        candidates.append(matches[-1])
    candidates.append(text)
    for candidate in candidates:
        normalized = _normalize_choice_label(candidate, allowed_letters)
        if normalized:
            return normalized
    return _normalize_choice_label(text.upper(), allowed_letters)


@registry.asset(
    "metrics",
    "multi_choice_accuracy",
    desc="Multiple-choice accuracy (llm-eval compatible)",
    tags=("text", "multiple-choice"),
    default_aggregation="mean",
)
class MultiChoiceAccuracyMetric(SimpleMetric):
    """Compares extracted option letters with reference answers."""

    value_key = "acc"

    def compute(self, context: MetricContext) -> MetricResult:
        score, prediction, expected = self._evaluate(context)
        metadata = {}
        if prediction is not None:
            metadata["prediction"] = prediction
        if expected is not None:
            metadata["expected"] = expected
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)

    def _evaluate(self, context: MetricContext) -> Tuple[float, Optional[str], Optional[str]]:
        sample_dict = extract_field(context, 'sample')
        expected_raw = get_first_reference(sample_dict)
        allowed_letters = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        expected = _normalize_choice_label(expected_raw, allowed_letters)        

        prediction_raw = get_text_content_of_first_predict_result(sample_dict)

        prediction = _extract_prediction_letter(prediction_raw, allowed_letters)

        if not expected or not prediction:
            return 0.0, prediction, expected
        return (1.0 if prediction == expected else 0.0), prediction, expected


__all__ = ["MultiChoiceAccuracyMetric"]
