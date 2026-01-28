"""FizBinBench eval_multi_round_desc accuracy metric"""

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
    "fin_multi_round_desc",
    desc="eval_multi_round_desc",
    tags=("BizFinBench V2",),
    default_aggregation="mean",
)
class FinMultiRoundDescAccuracyMetric(SimpleMetric):
    value_key = "acc"
    def compute(self, context: MetricContext) -> MetricResult:
        score = 0
        metadata = {}
        false_desc_id = []        
        # STEP 1: extract sample/predict /groud truth
        try:
            sample_dict = extract_field(context, 'sample')
            correct_answer = get_first_reference(sample_dict)
            predict_result_str = get_text_content_of_first_predict_result(sample_dict)  
            predicted_answer = None
            try:
                predict_data = re.sub('[\s\S]*think>', '', predict_result_str).strip()
            except json.JSONDecodeError:
                print("JSONDecodeError")
            desc_key = 'description_id' if 'description_id' in correct_answer else '描述编号'
            # extract ground truth and predict
            #print("correct_answer", correct_answer)
            #print("pred", predict_result_str, predict_data)
            correct_answer = json.loads(correct_answer)
            correct_answer_dict = {item[desc_key]: item['answer'] for item in correct_answer}
            # extract number
            total_answer = len(correct_answer)
            correct_count = 0

    
    
            # filter predict_data 1.```json[]``` 2.```[]``` 3. [] 
            format_one = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", predict_data, re.IGNORECASE)
            if format_one:
                predict_data = format_one.group(1).strip()
            else:
                format_two = re.findall(r"\[\s*\{[\s\S]*\}\s*\]", predict_data, re.IGNORECASE)
                if format_two:
                    predict_data = format_two[-1]
            predicted_answer_list = json.loads(predict_data)
            # print(predicted_answer_list)
            for predicted_item in predicted_answer_list:
                desc_id = predicted_item[desc_key]
                predicted_answer = sorted([int(re.search('\d+', str(ans)).group()) for ans in predicted_item['answer']])
        
                # check whether description_id in ground truth
                if desc_id in correct_answer_dict:
                    if predicted_answer == sorted(correct_answer_dict[desc_id]):
                        correct_count += 1
                    else:
                        false_desc_id.append(desc_id)
    
            score = correct_count / total_answer
            if score == 1:
                result = "True"
            elif 0 < float(score) < 1:
                result = "Partially Correct"
            else:
                result = "False"
            false_desc_ids = ','.join(str(x) for x in false_desc_id)
            metadata['eval_result'] = {
                "result": result,
                "false_desc_id": false_desc_ids
            }                   
        except Exception as e:
            score = 0
            result = "False"
            metadata['eval_result'] = {"result": "False", "error": str(e)}        

        # STEP 3: compute score
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)

__all__ = ["FinMultiRoundDescAccuracyMetric", ]

if __name__ ==  '__main__':
    from gage_eval.config.pipeline_config import MetricSpec
    spec = MetricSpec(metric_id='test', implementation='fake_acc')
    global_metric = FinMultiRoundDescAccuracyMetric(spec=spec)