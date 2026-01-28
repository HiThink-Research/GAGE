"""FizBinBench eval_financial_description_no_cot accuracy metric"""

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
import json
import re
from typing import Optional, Tuple, Dict, Any

from gage_eval.metrics.builtin.biz_fin_bench_v2.utils import JsonPaser

@registry.asset(
    "metrics",
    "fin_desc",
    desc="eval_financial_description_no_cot",
    tags=("BizFinBench V2",),
    default_aggregation="mean",
)
class FinDescAccuracyMetric(SimpleMetric):
    value_key = "acc"
    def compute(self, context: MetricContext) -> MetricResult:
        score = 0
        metadata = {}        
        try:
            # STEP 1: extract sample/predict /groud truth
            sample_dict = extract_field(context, 'sample')
            correct_answers = get_first_reference(sample_dict)
            correct_answers = json.loads(correct_answers)
            if not isinstance(correct_answers, list):
                correct_answers = []

            predict_result_str = get_text_content_of_first_predict_result(sample_dict)        

            j_paser = JsonPaser()
            predict_data = j_paser.extract_json_from_text(predict_result_str)

            if predict_data and isinstance(predict_data, dict):
                predicted_answers = predict_data["answer"] if predict_data.get("answer") else []
            else:
                predicted_answers = []

            if not isinstance(predicted_answers, list):
                predicted_answers = []

            predicted_answers = [int(x) for x in predicted_answers]


            if sorted(predicted_answers) == sorted(correct_answers):
                score = 1.0
                metadata['eval_result'] = {"result": "True"}
            else:
                score = 0.0
                metadata['eval_result'] = {"result": "False"}
            metadata.update({"prediction": predicted_answers, "references": correct_answers})    
        except Exception as e:
            metadata['eval_result'] = {"result": f"error: {str(e)}"}
            
        # STEP 3: compute score    
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)

__all__ = ["FinDescAccuracyMetric", ]

if __name__ ==  '__main__':
    from gage_eval.config.pipeline_config import MetricSpec
    spec = MetricSpec(metric_id='test', implementation='fake_acc')
    global_metric = FinDescAccuracyMetric(spec=spec)