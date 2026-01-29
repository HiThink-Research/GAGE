from __future__ import annotations

from gage_eval.assets.datasets.preprocessors.live_code_bench.lm_styles import LanguageModelStore, LMStyle, LanguageModel
from gage_eval.assets.datasets.loaders.live_code_bench.scenarios import Scenario

from gage_eval.metrics.builtin.live_code_bench.utils.extraction_utils import (
    extract_code,
    extract_test_output_code,
    extract_execution_code,
)

def combine_results(
    scenario: Scenario,
    results: list[list[str]],
    model: LanguageModel,
    cot_code_execution: bool = False,
):
    if scenario == Scenario.codegeneration:
        combined_results = [
            (
                outputs_list,
                [extract_code(output, model.model_style) for output in outputs_list],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.testoutputprediction:
        combined_results = [
            (
                outputs_list,
                [
                    extract_test_output_code(output, model.model_style)
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.selfrepair:
        combined_results = [
            (
                [
                    output[0] if type(output) is list else output
                    for output in outputs_list
                ],
                [
                    (
                        extract_code(output[0], model.model_style)
                        if type(output) is list
                        else extract_code(output, model.model_style)
                    )
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.codeexecution:
        combined_results = [
            (
                outputs_list,
                [
                    extract_execution_code(
                        output, model.model_style, cot=cot_code_execution
                    )
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    else:
        raise ValueError(f"Scenario {scenario} not implemented")

    return combined_results
