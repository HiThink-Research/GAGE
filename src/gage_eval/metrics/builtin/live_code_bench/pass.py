""" Live Code Bench pass metric"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Literal

from gage_eval.assets.datasets.preprocessors.live_code_bench.lm_styles import LanguageModelStore, LMStyle, LanguageModel
from gage_eval.assets.datasets.loaders.live_code_bench.scenarios import Scenario

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

def get_results(sample_dict):
    results = []
    try:
        this_pred = []
        predict_result = sample_dict['predict_result']
        for pred in predict_result:
            this_pred.append(pred['message']['content'][0]['text'])
        results.append(this_pred)
    except Exception as e:
        logger.warning(f'[warning]{e}')        
        return results
    return results


@registry.asset(
    "metrics",
    "live_code_bench_pass", 
    desc="live_code_bench pass@K",
    tags=("live_code_bench", "pass@K"),
    default_aggregation="mean",
)
class LiveCodeBenchPassMetric(SimpleMetric):
    value_key = "pass@K"
    regex_pattern =  r"the answer is \s*(.+?)\s*\r?\n\s*"
    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: extract sample/predict /groud truth
        sample_dict = extract_field(context, 'sample')
        metadata = sample_dict.get('metadata')
        model_dict = metadata.get('model')
        _scenario_str = metadata.get('scenario')
        scenario = Scenario(_scenario_str)

        model = LanguageModel.from_dict(model_dict)
        print("sample_dict", sample_dict)
        results = get_results(sample_dict)
        print("results:", results)
  

        # STEP 3: compute score
        final_pred, score = match_str(pred, str(answer), location="exact")
        score = float(score)
        metadata = {"prediction": final_pred, "references": answer}
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)

__all__ = ["LiveCodeBenchPassMetric", ]

if __name__ ==  '__main__':
    from gage_eval.config.pipeline_config import MetricSpec
    spec = MetricSpec(metric_id='test', implementation='fake_acc')
    live_code_metric = LiveCodeBenchPassMetric(spec=spec)
