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

def extract_numeric_value(text):
    """
    extract numeric value
    """
    if text is None:
        return None
    

    pattern = r'-?\d+\.?\d*|\.\d+'
    matches = re.findall(pattern, str(text))
    
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None

@registry.asset(
    "metrics",
    "fin_cal_2",
    desc="eval_financial_calculation_2",
    tags=("BizFinBench V2",),
    default_aggregation="mean",
)
class FinCal2AccuracyMetric(SimpleMetric):
    value_key = "acc"
    def compute(self, context: MetricContext) -> MetricResult:
        score = 0
        # STEP 1: extract sample/predict /groud truth
        sample_dict = extract_field(context, 'sample')
        correct_answer = get_first_reference(sample_dict)
        predicted_answer = None
        predict_result_str = get_text_content_of_first_predict_result(sample_dict)        

        try:
            j_paser = JsonPaser()
            
            predict_data = j_paser.extract_json_from_text(predict_result_str)

            if predict_data:
                predicted_answer = predict_data.get("answer", "")           
        except json.JSONDecodeError:
            try:
                json_pattern = r'"answer"\s*:\s*([^,}\s]+)'
                match = re.search(json_pattern, predict_result_str)
                if match:
                    predicted_answer_str = match.group(1)
                    if predicted_answer_str.startswith('"') and predicted_answer_str.endswith('"'):
                        predicted_answer = predicted_answer_str[1:-1]
                    else:
                        predicted_answer = predicted_answer_str
            except Exception as e:
                print(f"Error extracting answer with regex: {e}")

        # extract number
        correct_numeric = extract_numeric_value(correct_answer)
        predicted_numeric = extract_numeric_value(predicted_answer)
        
        # compare
        if correct_numeric is not None and predicted_numeric is not None:
            score = 1.0 if abs(predicted_numeric - correct_numeric) < 1e-4 else 0
        else:
            score = 1.0 if str(predicted_answer).strip() == str(correct_answer).strip() else 0

        # STEP 3: compute score
        metadata = {"prediction": predicted_answer, "references": correct_answer}
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)

__all__ = ["FinCal2AccuracyMetric", ]

if __name__ ==  '__main__':
    from gage_eval.config.pipeline_config import MetricSpec
    spec = MetricSpec(metric_id='test', implementation='fake_acc')
    global_metric = FinCal2AccuracyMetric(spec=spec)
