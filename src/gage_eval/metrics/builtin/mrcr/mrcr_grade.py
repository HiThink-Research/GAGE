""" MRCR grade metric (Local)"""

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
from difflib import SequenceMatcher

@registry.asset(
    "metrics",
    "mrcr_grade", 
    desc="MRCR grade",
    tags=("MRCR",),
    default_aggregation="mean",
)
class MRCRGradeMetric(SimpleMetric):
    value_key = "grade"
    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: extract sample/predict /groud truth
        sample_dict = extract_field(context, 'sample')
        answer = get_first_reference(sample_dict)     
        response = get_text_content_of_first_predict_result(sample_dict)
        metadata = sample_dict.get("metadata")
        random_string_to_prepend = None
        if metadata:
            random_string_to_prepend = metadata.get("random_string_to_prepend") 
        # STEP 2: pretty prediction
        if not response.startswith(random_string_to_prepend):
            score = 0.0
        else:
            response = response.removeprefix(random_string_to_prepend)
            answer = answer.removeprefix(random_string_to_prepend)
            score = float(SequenceMatcher(None, response, answer).ratio())

        # STEP 3: compute score
        metadata = {"prediction": response, "references": answer}
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)

__all__ = ["MRCRGradeMetric", ]

if __name__ ==  '__main__':
    from gage_eval.config.pipeline_config import MetricSpec
    spec = MetricSpec(metric_id='test', implementation='fake_acc')
    hle_metric = MRCRGradeMetric(spec=spec)
