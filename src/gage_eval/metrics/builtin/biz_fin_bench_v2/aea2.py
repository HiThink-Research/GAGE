"""FizBinBench AEA_2 accuracy metric"""

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
    "aea2",
    desc="AEA_2",
    tags=("BizFinBench V2",),
    default_aggregation="mean",
)
class AEA2AccuracyMetric(SimpleMetric):
    value_key = "acc"
    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: extract sample/predict /groud truth
        sample_dict = extract_field(context, 'sample')
        correct_answers = get_first_reference(sample_dict)
        correct_answers = [int(x) for x in correct_answers.split(',')]
        if not isinstance(correct_answers, list):
            correct_answers = []

        predict_result_str = get_text_content_of_first_predict_result(sample_dict)
        
        j_paser = JsonPaser()
            
        predict_data = j_paser.extract_json_from_text(predict_result_str)
        # if predict_data:
        if predict_data and isinstance(predict_data, dict):
            if predict_data.get("相关新闻序号", ""):
                predicted_answers = predict_data["相关新闻序号"]
            elif any('序号' in key for key in predict_data):
                predicted_answers = predict_data["相关内容序号"] if predict_data.get("相关内容序号", "") else predict_data.get("无关内容序号", "")
            else:
                predicted_answers = predict_data["Relevant Content Numbers"] if predict_data.get("Relevant Content Numbers", "") else predict_data.get("Irrelevant Content Numbers", "")
        else:
            predicted_answers = []

        if not isinstance(predicted_answers, list):
            predicted_answers = []

        predicted_answers = [int(x) for x in predicted_answers]
        
        metadata = {}
        if sorted(predicted_answers) == sorted(correct_answers):
            score = 1.0
            metadata['eval_result'] = {"result": "True"}
        elif predicted_answers != [] and set(predicted_answers).issubset(set(correct_answers)):
            score = round(len(predicted_answers)/len(correct_answers), 2)
            metadata['eval_result'] = {"result": "Partially Correct"}
        else:
            score = 0.0
            metadata['eval_result'] = {"result": "False"}
 
        # STEP 3: compute score
        metadata.update({"prediction": predicted_answers, "references": correct_answers})
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)

__all__ = ["AEA2AccuracyMetric", ]

if __name__ ==  '__main__':
    from gage_eval.config.pipeline_config import MetricSpec
    spec = MetricSpec(metric_id='test', implementation='fake_acc')
    global_metric = AEA2AccuracyMetric(spec=spec)
