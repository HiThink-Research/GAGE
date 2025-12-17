"""DocVQA ANLS metric mirroring llm-eval's implementation."""

from __future__ import annotations

from typing import Any

from gage_eval.metrics.base import MultiReferenceTextMetric
from gage_eval.metrics.utils import levenshtein_distance, strip_thought_tags
from gage_eval.registry import registry


@registry.asset(
    "metrics",
    "docvqa_anls",
    desc="DocVQA Average Normalized Levenshtein Similarity",
    tags=("vision", "vqa", "docvqa"),
    default_aggregation="mean",
)
class DocVQAANLSMetric(MultiReferenceTextMetric):
    """Computes ANLS between prediction and any of the reference answers."""

    value_key = "anls"

    default_prediction_field = "model_output.answer"
    default_reference_field = "sample.metadata.answers"

    def _normalize_prediction(self, prediction: Any) -> str:
        stripped = _strip_reasoning(prediction)
        return super()._normalize_prediction(stripped)

    def score_single(self, prediction: str, reference: str) -> float:
        if not prediction and not reference:
            return 1.0
        if not prediction or not reference:
            return 0.0
        distance = levenshtein_distance(prediction, reference)
        denom = max(len(prediction), len(reference)) or 1
        return 1.0 - (distance / float(denom))

    def aggregate_scores(self, scores, references, prediction: str) -> tuple[float, dict]:
        scores_list = list(scores)
        if not scores_list:
            return 0.0, {"warning": "empty_refs"}
        best_score = max(scores_list)
        best_ref = references[scores_list.index(best_score)] if references else None
        threshold = float(self.args.get("threshold", 0.5))
        final_score = best_score if best_score >= threshold else 0.0
        return final_score, {"best_score": best_score, "best_reference": best_ref, "threshold": threshold}


def _strip_reasoning(text: Any) -> str:
    return strip_thought_tags(text)


__all__ = ["DocVQAANLSMetric"]
