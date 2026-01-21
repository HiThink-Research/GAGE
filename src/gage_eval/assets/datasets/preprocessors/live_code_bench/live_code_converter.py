"""LiveCodeBench preprocessors."""

from __future__ import annotations

import random
from typing import Any, Dict
import json


from loguru import logger

from gage_eval.assets.datasets.utils.mapping import (
    extract_field,
    normalize_options,
    resolve_correct_choice,
)

from gage_eval.assets.datasets.sample import (
    SCHEMA_VERSION,
    Sample,
    Message,
    MessageContent,
    sample_from_dict
)

from dataclasses import asdict

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.loaders.live_code_bench.scenarios import Scenario
from gage_eval.assets.datasets.preprocessors.live_code_bench.code_generation import format_prompt_generation
from gage_eval.assets.datasets.preprocessors.live_code_bench.lm_styles import LanguageModelStore
from gage_eval.assets.datasets.preprocessors.live_code_bench.utils import normalize_messages
from enum import Enum
from typing import Any, Dict, List, Union

def convert_enum_to_str(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return str(obj)          # 或 obj.value，看你要的是 name 还是 value
    elif isinstance(obj, dict):
        return {k: convert_enum_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_enum_to_str(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_enum_to_str(item) for item in obj)
    else:
        return obj

def normalize_tests(test_list):
    ret_list = []
    for d in test_list:
        tmp = convert_enum_to_str(d)
        ret_list.append(tmp)
    return ret_list
    

class LiveCodeBenchConverter(BasePreprocessor):
    """Preprocesses LiveCodeBench records into the Sample schema."""

    
    def to_sample(self, record: Dict[str, Any], schema_version = SCHEMA_VERSION, 
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct",
        scenario=None, **kwargs: Any) -> Sample:
        """Converts a raw LiveCodeBench record into a standardized Sample.

        Args:
            record: Raw dataset record (typically a dict emitted by the loader).
            **kwargs: Reserved for forward compatibility.

        Returns:
            A Sample with the gage-eval Sample schema.
        """
        sample: Dict[str, Any] = dict(record)
        _scenario_str = scenario or sample.get('scenario')

        scenario = Scenario(_scenario_str)

        model = LanguageModelStore[model_name]

        if scenario == Scenario.codegeneration:
            dict_messages = format_prompt_generation(sample, model.model_style)
            final_messages = normalize_messages(dict_messages)


        sample_id = '_'.join([sample.get('question_id'), sample.get("contest_id")])
        ref = [
            {k: normalize_tests(sample[k])} for k in ['public_test_cases', 'private_test_cases']
        ]
        self_metadata = str(sample.get('metadata')) or '{}'
        metadata = {
            'scenario': _scenario_str,
            'metadata': self_metadata,
            'model': model.to_dict()
        }
        ret_sample = Sample(
            id = sample_id,
            schema_version = schema_version,
            messages = final_messages,
            references = ref,
            metadata = metadata
        )
        # print("ret_sample:", ret_sample)
        return ret_sample


if __name__ == '__main__':
    sample = {}
    pre = LiveCodeBenchConverter()    
    ret = pre.to_sample(sample)
  