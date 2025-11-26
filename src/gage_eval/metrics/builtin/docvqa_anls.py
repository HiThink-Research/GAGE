"""DocVQA ANLS metric mirroring llm-eval's implementation."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, List, Optional, Sequence

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.registry import registry

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@registry.asset(
    "metrics",
    "docvqa_anls",
    desc="DocVQA Average Normalized Levenshtein Similarity",
    tags=("vision", "vqa", "docvqa"),
    default_aggregation="mean",
)
class DocVQAANLSMetric(SimpleMetric):
    """Computes ANLS between prediction and any of the reference answers."""

    value_key = "anls"

    def compute(self, context: MetricContext) -> MetricResult:
        answers_field = self.args.get("answers_field", "sample.metadata.answers")
        prediction_field = self.args.get("prediction_field", "model_output.answer")
        threshold = float(self.args.get("threshold", 0.5))

        answers_raw = _extract_field(context, answers_field, default=())
        answers = _coerce_string_list(answers_raw)
        prediction_raw = _extract_field(context, prediction_field, default="")
        prediction = _normalize_text(_strip_reasoning(prediction_raw))

        score = 0.0
        best_match = None
        if answers and prediction:
            similarities = [_anls(answer, prediction) for answer in answers]
            best_score = max(similarities) if similarities else 0.0
            if best_score >= threshold:
                score = best_score
                best_match = answers[similarities.index(best_score)]

        metadata = {
            "prediction": prediction,
            "best_match": best_match,
            "threshold": threshold,
        }
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)


def _extract_field(context: MetricContext, descriptor: Optional[str], default: Any = None) -> Any:
    if not descriptor:
        return default
    roots = {
        "sample": context.sample,
        "model_output": context.model_output,
        "judge_output": context.judge_output,
    }
    parts = descriptor.split(".")
    root_key = parts[0]
    base = roots.get(root_key, context.sample)
    tail = ".".join(parts[1:]) if len(parts) > 1 else ""
    if not tail:
        return base
    current = base
    for part in tail.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def _coerce_string_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, Sequence):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [str(raw).strip()]


def _strip_reasoning(text: Any) -> str:
    raw = "" if text is None else str(text)
    return _THINK_PATTERN.sub("", raw).strip()


def _normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    return re.sub(r"\s+", " ", lowered)


def _anls(reference: str, prediction: str) -> float:
    ref = _normalize_text(reference)
    pred = _normalize_text(prediction)
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    distance = _levenshtein_distance(ref, pred)
    length = max(len(ref), len(pred))
    if length == 0:
        return 1.0
    return 1.0 - (distance / float(length))


def _levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


__all__ = ["DocVQAANLSMetric"]
