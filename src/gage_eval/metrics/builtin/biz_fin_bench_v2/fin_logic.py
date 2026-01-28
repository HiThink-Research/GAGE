"""FizBinBench eval_financial_logic_event accuracy metric"""

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
    "fin_logic",
    desc="eval_financial_logic_event2",
    tags=("BizFinBench V2",),
    default_aggregation="mean",
)
class FinLogicAccuracyMetric(SimpleMetric):
    value_key = "acc"
    def compute(self, context: MetricContext) -> MetricResult:
        score = 0
        metadata = {}        
        try:
            # STEP 1: extract sample/predict /groud truth
            sample_dict = extract_field(context, 'sample')
            correct_answer = get_first_reference(sample_dict)

            predict_result_str = get_text_content_of_first_predict_result(sample_dict)        

            j_paser = JsonPaser()
            predict_data = j_paser.extract_json_from_text(predict_result_str)

            # extract answer
            predicted_answer = ""

            if predict_data:
                # try to extract
                if "answer" in predict_data:
                    predicted_answer = str(predict_data["answer"]).strip()
                elif "排序结果" in predict_data:  # cn key
                    predicted_answer = str(predict_data["排序结果"]).strip()
                elif "result" in predict_data:
                    predicted_answer = str(predict_data["result"]).strip()
            
                # if failed, match str
                if not predicted_answer and "answer" in predict_result_str:
                    match = re.search(r'"answer"\s*:\s*"([^"]+)"', predict_result_str)
                    if match:
                        predicted_answer = match.group(1).strip()           
            # compare
            correct_normalized = re.sub(r'\s+', '', correct_answer)
            predicted_normalized = re.sub(r'\s+', '', predicted_answer)

            if correct_normalized == predicted_normalized:
                score = 1.0
                metadata['eval_result'] = {"result": "True"}
            else:
                score = 0.0
                metadata['eval_result'] = {
                    "result": "False", 
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer
                }
        except Exception as e:
            score = 0
            metadata['eval_result'] = {"result": f"error: {str(e)}"}
        # STEP 3: compute score
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)

__all__ = ["FinLogicAccuracyMetric", ]

if __name__ ==  '__main__':
    from gage_eval.config.pipeline_config import MetricSpec
    spec = MetricSpec(metric_id='test', implementation='fake_acc')
    global_metric = FinLogicAccuracyMetric(spec=spec)
