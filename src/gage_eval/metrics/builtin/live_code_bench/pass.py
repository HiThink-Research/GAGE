""" Live Code Bench pass metric"""

from __future__ import annotations

import re
import json
from typing import Any, Dict, Optional, Literal

from gage_eval.assets.datasets.preprocessors.live_code_bench.lm_styles import LanguageModelStore, LMStyle, LanguageModel
from gage_eval.assets.datasets.loaders.live_code_bench.scenarios import Scenario
from gage_eval.metrics.builtin.live_code_bench.evaluation.compute_code_generation_metrics import codegen_metrics
from gage_eval.metrics.builtin.live_code_bench.evaluation.compute_code_execution_metrics import code_execution_metrics
# test output:
from gage_eval.metrics.builtin.live_code_bench.evaluation.compute_test_output_prediction_metrics import (
    test_output_metrics,
)
from gage_eval.assets.datasets.preprocessors.live_code_bench.test_output_prediction import (
    TestOutputPredictionProblem, 
)

# code exec:
from gage_eval.metrics.builtin.live_code_bench.evaluation.compute_code_execution_metrics import (
    code_execution_metrics,
)
from gage_eval.assets.datasets.preprocessors.live_code_bench.code_execution import (
    CodeExecutionProblem,
)

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric

from gage_eval.metrics.filter.base import RegexFilter
from gage_eval.metrics.numeric import extract_numeric_answer
from gage_eval.metrics.match import match_str
from gage_eval.metrics.choice import extract_single_choice_letter
from gage_eval.metrics.builtin.live_code_bench.router import combine_results

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

def get_eval_sample(ref_list, fn_name=None):
    return {
        "input_output": json.dumps(
            {
                "inputs": [
                    t.get('input') for t in ref_list
                ],
                "outputs":[
                    t.get('output') for t in ref_list
                ],
                "fn_name": fn_name
            }
        )
    }


@registry.asset(
    "metrics",
    "live_code_bench_pass", 
    desc="live_code_bench pass@K",
    tags=("live_code_bench", "pass@K"),
    default_aggregation="mean",
)
class LiveCodeBenchPassMetric(SimpleMetric):
    value_key = "pass@1"
    def codegen(self, sample_id, sample_dict, metadata, k_list, scenario, cot_code_execution):
        fn_name = None
        dataset_meta =  metadata.get('metadata')
        if dataset_meta is not None:
            try:
                meta = json.loads(dataset_meta)
                fn_name = meta.get("func_name")
            except:
                fn_name = None
        model_dict = metadata.get('model')
        ref = sample_dict.get("references")
        model = LanguageModel.from_dict(model_dict)
        #print("sample_dict", sample_dict)
        results = get_results(sample_dict)
        combined_results = combine_results(scenario, results, model, cot_code_execution)
        eval_samples = [get_eval_sample(ref, fn_name)]
        generations = [extracted for _, extracted in combined_results]
        try:
            metrics = codegen_metrics(
                eval_samples,
                generations,
                num_process_evaluate=5,
                timeout=timeout,
            )
        except:
            pass
        #print("ref:", ref)
        #print("results:", results)
        #print("self.spec", self.spec)
        #print("sef.args", self.args)
        #print("eval_samples", eval_samples)
        #print("generation:", generations)

        #exit(0)

        return self.make_ret(sample_id, scenario, k_list, metrics)

    def test_output(self, sample_id, sample_dict, metadata, k_list, scenario, cot_code_execution):
        model_dict = metadata.get('model')
        ref = sample_dict.get("references")[0]
        model = LanguageModel.from_dict(model_dict)
        #print("sample_dict", sample_dict)
        results = get_results(sample_dict)
        combined_results = combine_results(scenario, results, model, cot_code_execution)
        benchmark = [TestOutputPredictionProblem(**ref)]
        eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
        generations = [extracted for _, extracted in combined_results]

        metrics = test_output_metrics(
            eval_samples,
            generations,
            k_list=k_list,
        )
        return self.make_ret(sample_id, scenario, k_list, metrics)

    def code_execution(self, sample_id, sample_dict, metadata, k_list, scenario, cot_code_execution):
        model_dict = metadata.get('model')
        ref = sample_dict.get("references")[0]
        model = LanguageModel.from_dict(model_dict)
        #print("sample_dict", sample_dict)
        results = get_results(sample_dict)
        combined_results = combine_results(scenario, results, model, cot_code_execution)
        benchmark = [CodeExecutionProblem(**ref)]
        eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
        generations = [extracted for _, extracted in combined_results]

        metrics = test_output_metrics(
            eval_samples,
            generations,
            k_list=k_list,
        )
        return self.make_ret(sample_id, scenario, k_list, metrics)
    def make_ret(self, sample_id, scenario, k_list, metrics):
        metadata = {"scenario": scenario.value}
        pass_key = [
            f'pass@{k}' for k in k_list
        ]
        values = {
            key: metrics[0][key] for key in pass_key
        }
        return MetricResult(sample_id=sample_id, values=values, metadata=metadata)            
    def compute(self, context: MetricContext) -> MetricResult:
        k_list = self.args.get("ks") or [1]
        timeout = self.args.get('timeout') or 6        
        # STEP 1: extract sample/predict /groud truth
        sample_dict = extract_field(context, 'sample')
        metadata = sample_dict.get('metadata')
        cot_code_execution = metadata.get("cot_code_execution") or False
        _scenario_str = metadata.get('scenario')
        scenario = Scenario(_scenario_str)
        if scenario == Scenario.codegeneration:
            return self.codegen(context.sample_id, sample_dict, metadata, k_list, scenario, cot_code_execution)
        elif scenario == Scenario.testoutputprediction:
            return self.test_output(context.sample_id, sample_dict, metadata, k_list, scenario, cot_code_execution)
        elif scenario == Scenario.codeexecution:
            return self.code_execution(context.sample_id, sample_dict, metadata, k_list, scenario, cot_code_execution)                

__all__ = ["LiveCodeBenchPassMetric", ]

if __name__ ==  '__main__':
    from gage_eval.config.pipeline_config import MetricSpec
    spec = MetricSpec(metric_id='test', implementation='fake_acc')
    live_code_metric = LiveCodeBenchPassMetric(spec=spec)
