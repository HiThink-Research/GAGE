"""MMMU-specific accuracy metric aligned with the legacy llm-eval script."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.metrics.utils import ensure_list_of_strings, extract_field, normalize_text_advanced
from gage_eval.registry import registry


def _is_choice_letter(value: str) -> bool:
    text = value.strip()
    return len(text) == 1 and text.isalpha()


def _match_prediction(prediction: str, target: str) -> bool:
    if not prediction or not target:
        return False
    pred = prediction.strip()
    tgt = target.strip()
    if not tgt:
        return False

    # NOTE: Prefer extracting the final single-letter choice to avoid false
    # positives from strings like "Option C:".
    if _is_choice_letter(tgt):
        extracted = _extract_choice_letter(pred)
        if extracted:
            return extracted.upper() == tgt.upper()
        # If no explicit choice letter is found, avoid substring matching and
        # fall back to strict comparison.
        return pred.upper() == tgt.upper()

    # NOTE: For non-letter targets, keep the relaxed matching behavior.
    if pred == tgt:
        return True
    if tgt in pred:
        return True
    if tgt.lower() in pred.lower():
        return True
    return False


def _extract_choice_letter(prediction: str) -> Optional[str]:
    """Extracts the last explicit multiple-choice letter from a model output.

    Supported patterns:
    - \\boxed{B}
    - <answer> B
    - standalone single-letter A-E (take the last occurrence)
    """

    import re

    patterns = [
        r"\\boxed\{\s*([A-Ea-e])\s*\}",
        r"<answer>\s*([A-Ea-e])",
        r"\b([A-Ea-e])\b",
    ]
    candidates: list[str] = []
    for pat in patterns:
        for match in re.findall(pat, prediction):
            if isinstance(match, tuple):
                for token in match:
                    if token:
                        candidates.append(token)
            elif match:
                candidates.append(match)
    return candidates[-1] if candidates else None


@registry.asset(
    "metrics",
    "mmmu_accuracy",
    desc="MMMU multimodal exact-match metric (llm-eval compatible)",
    tags=("vision", "mmmu"),
    default_aggregation="mean",
)
class MMMUAccuracyMetric(SimpleMetric):
    value_key = "acc"

    def compute(self, context: MetricContext) -> MetricResult:
        label_field = self.args.get("label_field", "sample.choices.0.message.content.0.text")
        prediction_field = self.args.get("prediction_field", "model_output.answer")

        raw_targets = extract_field(context, label_field, default="")
        targets = [
            normalized
            for normalized in (
                normalize_text_advanced(text, strip=True, collapse_whitespace=True)
                for text in ensure_list_of_strings(raw_targets)
            )
            if normalized
        ]
        prediction_raw = extract_field(context, prediction_field, default="")
        prediction = normalize_text_advanced(str(prediction_raw), strip=True, collapse_whitespace=True) or ""

        matched = any(_match_prediction(prediction, target) for target in targets)
        score = 1.0 if matched else 0.0
        metadata = {
            "prediction": prediction,
            "expected": targets,
        }
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)


__all__ = ["MMMUAccuracyMetric"]
