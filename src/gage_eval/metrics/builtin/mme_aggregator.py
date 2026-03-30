"""MME-specific aggregator for computing acc_plus (per-image accuracy).

This aggregator groups samples by question_id and computes:
- acc: per-question accuracy
- acc_plus: per-image accuracy (both questions correct)

Reference: Awesome-Multimodal-Large-Language-Models/eval_tool/calculation.py
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

from loguru import logger
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.aggregators import MetricAggregator
from gage_eval.metrics.base import AggregatedMetric, MetricResult
from gage_eval.metrics.runtime_context import AggregationRuntimeContext


class MMEAccPlusAggregator(MetricAggregator):
    """MME-specific aggregator that computes acc_plus (per-image accuracy).
    
    MME dataset: one image (question_id) corresponds to two questions.
    acc_plus: percentage of images where both questions are answered correctly.
    
    Reference: Awesome-Multimodal-Large-Language-Models/eval_tool/calculation.py
    """

    def __init__(
        self,
        spec: MetricSpec,
        runtime_context: Optional[AggregationRuntimeContext] = None,
    ) -> None:
        super().__init__(spec, runtime_context=runtime_context)
        self._group_field = str(spec.params.get("group_field", "question_id"))
        self._value_key = str(spec.params.get("value_key", "acc"))
        self._question_groups: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "correct": 0}
        )
        self._total_questions = 0
        self._correct_questions = 0
        self._missing_group_id_samples = 0

    def add(self, result: MetricResult) -> None:
        self._total_questions += 1
        is_correct = float(result.values.get(self._value_key, 0.0)) == 1.0
        if is_correct:
            self._correct_questions += 1
        question_id = result.metadata.get(self._group_field)
        if not question_id:
            self._missing_group_id_samples += 1
            return
        stats = self._question_groups[str(question_id)]
        stats["total"] += 1
        if is_correct:
            stats["correct"] += 1

    def finalize(self) -> AggregatedMetric:
        # STEP 2: Calculate acc_plus (per-image accuracy)
        # For each question_id group, check if both questions are correct
        total_images = 0
        correct_images = 0

        for question_id, stats in self._question_groups.items():
            if stats["total"] < 2:
                continue
            if stats["total"] > 2:
                logger.warning(
                    "MMEAccPlusAggregator: Found {} samples for question_id={}, expected 2",
                    stats["total"],
                    question_id,
                )
            total_images += 1
            if stats["correct"] == stats["total"]:
                correct_images += 1

        # STEP 3: Calculate metrics (like calculation.py)
        # acc: per-question accuracy (all samples)
        acc = (
            self._correct_questions / self._total_questions
            if self._total_questions > 0
            else 0.0
        )
        # acc_plus: per-image accuracy (both questions correct)
        acc_plus = correct_images / total_images if total_images > 0 else 0.0

        logger.debug(
            "MMEAccPlusAggregator finalized metric={} questions={} correct_questions={} images={} correct_images={} acc={:.4f} acc_plus={:.4f}",
            self.spec.metric_id,
            self._total_questions,
            self._correct_questions,
            total_images,
            correct_images,
            acc,
            acc_plus,
        )

        return AggregatedMetric(
            metric_id=self.spec.metric_id,
            aggregation=self.spec.aggregation or "mme_acc_plus",
            values={
                "acc": acc,
                "acc_plus": acc_plus,
            },
            count=self._total_questions,
            metadata={
                "total_images": total_images,
                "correct_images": correct_images,
                "total_questions": self._total_questions,
                "correct_questions": self._correct_questions,
                "group_count": len(self._question_groups),
                "missing_group_id_samples": self._missing_group_id_samples,
            },
        )


__all__ = ["MMEAccPlusAggregator"]
