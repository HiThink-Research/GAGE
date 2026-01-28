""" Global PIQA accuracy metric (Local)"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Literal

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric

from gage_eval.metrics.filter.base import RegexFilter
from gage_eval.metrics.numeric import extract_numeric_answer
from gage_eval.metrics.match import match_str
from gage_eval.metrics.choice import extract_single_choice_letter

from gage_eval.metrics.utils import (
    ensure_list_of_strings, extract_field, normalize_text_advanced,
        get_text_content_of_first_predict_result,
    get_sample_label,
    get_first_reference
)

from gage_eval.registry import registry

@registry.asset(
    "metrics",
    "global_piqa_accuracy_local", 
    desc="Global PIQA accuracy",
    tags=("Global PIQA",),
    default_aggregation="mean",
)
class GlobalPIQAAccuracyMetric(SimpleMetric):
    value_key = "acc"
    regex_pattern = r"The best answer is:\s*(.+?)"
    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: extract sample/predict /groud truth
        sample_dict = extract_field(context, 'sample')
        answer = get_first_reference(sample_dict)     
        prediction_raw = get_text_content_of_first_predict_result(sample_dict)


        # STEP 2: pretty prediction
        rf = RegexFilter(regex_pattern=self.regex_pattern, group_select=-1, ignore_case=True)
        pred = rf.apply(prediction_raw)
        pred = extract_single_choice_letter(pred, patterns = [
            r"\b([A-B])\b",
        ])

        # STEP 3: compute score
        final_pred, score = match_str(pred, str(answer), location="exact")
        score = float(score)
        metadata = {"prediction": final_pred, "references": answer}
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)

__all__ = ["GlobalAccuracyMetric", ]

if __name__ ==  '__main__':
    from gage_eval.config.pipeline_config import MetricSpec
    spec = MetricSpec(metric_id='test', implementation='fake_acc')
    global_metric = GlobalAccuracyMetric(spec=spec)
