"""MME-specific aggregator for computing acc_plus (per-image accuracy).

This aggregator groups samples by question_id and computes:
- acc: per-question accuracy
- acc_plus: per-image accuracy (both questions correct)

Reference: Awesome-Multimodal-Large-Language-Models/eval_tool/calculation.py
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from loguru import logger
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.aggregators import MetricAggregator
from gage_eval.metrics.base import AggregatedMetric, MetricResult


class MMEAccPlusAggregator(MetricAggregator):
    """MME-specific aggregator that computes acc_plus (per-image accuracy).
    
    MME dataset: one image (question_id) corresponds to two questions.
    acc_plus: percentage of images where both questions are answered correctly.
    
    Reference: Awesome-Multimodal-Large-Language-Models/eval_tool/calculation.py
    """

    def __init__(self, spec: MetricSpec) -> None:
        super().__init__(spec)
        self._results: List[MetricResult] = []
        self._group_field = str(spec.params.get("group_field", "question_id"))
        self._value_key = str(spec.params.get("value_key", "acc"))

    def add(self, result: MetricResult) -> None:
        self._results.append(result)

    def finalize(self) -> AggregatedMetric:
        # STEP 1: Group results by question_id (each image has two questions = two samples)
        # Each sample represents one question, and we need to group by question_id
        # to calculate acc_plus (both questions correct = image correct)
        question_id_groups: Dict[str, List[MetricResult]] = defaultdict(list)
        
        total_questions = 0
        correct_questions = 0
        
        for result in self._results:
            question_id = result.metadata.get("question_id")
            if question_id:
                question_id_groups[str(question_id)].append(result)
            else:
                # If no question_id, treat as standalone question
                question_id_groups[f"_standalone_{len(question_id_groups)}"].append(result)
            
            # Count per-question accuracy (acc)
            total_questions += 1
            if result.values.get(self._value_key, 0.0) == 1.0:
                correct_questions += 1
        
        # STEP 2: Calculate acc_plus (per-image accuracy)
        # For each question_id group, check if both questions are correct
        total_images = 0
        correct_images = 0
        
        for question_id, results in question_id_groups.items():
            if question_id.startswith("_standalone_"):
                # Standalone questions don't contribute to acc_plus
                continue
            
            # Each image should have exactly 2 questions (2 samples)
            if len(results) == 2:
                total_images += 1
                # Both questions must be correct for the image to be correct
                both_correct = all(
                    result.values.get(self._value_key, 0.0) == 1.0 
                    for result in results
                )
                if both_correct:
                    correct_images += 1
            elif len(results) > 2:
                # More than 2 questions for same image - log warning
                logger.warning(
                    f"MMEAccPlusAggregator: Found {len(results)} samples for question_id={question_id}, expected 2"
                )
                # Still count as one image, but require all to be correct
                total_images += 1
                all_correct = all(
                    result.values.get(self._value_key, 0.0) == 1.0 
                    for result in results
                )
                if all_correct:
                    correct_images += 1
            # If len(results) == 1, skip (incomplete image pair)

        # STEP 3: Calculate metrics (like calculation.py)
        # acc: per-question accuracy (all samples)
        acc = correct_questions / total_questions if total_questions > 0 else 0.0
        # acc_plus: per-image accuracy (both questions correct)
        acc_plus = correct_images / total_images if total_images > 0 else 0.0

        logger.debug(
            "MMEAccPlusAggregator finalized metric={} questions={} correct_questions={} images={} correct_images={} acc={:.4f} acc_plus={:.4f}",
            self.spec.metric_id,
            total_questions,
            correct_questions,
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
            count=total_questions,  # Count total questions (samples)
            metadata={
                "total_images": total_images,
                "correct_images": correct_images,
                "total_questions": total_questions,
                "correct_questions": correct_questions,
            },
        )


__all__ = ["MMEAccPlusAggregator"]
