"""BizFinBench v2 preprocessors."""

from __future__ import annotations

import random
from typing import Any, Dict
import transformers

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
from gage_eval.assets.datasets.preprocessors.live_code_bench.utils import normalize_messages


class BizFinBenchV2Converter(BasePreprocessor):
    """Preprocesses BizFinBenchV2 records into the Sample schema."""
    def to_sample(self, record: Dict[str, Any],
                  schema_version = SCHEMA_VERSION,
                  **kwargs: Any) -> Sample:
        """Converts a raw BizFinBenchV2 record into a standardized Sample.

        Args:
            record: Raw dataset record (typically a dict emitted by the loader).
            **kwargs: Reserved for forward compatibility.

        Returns:
            A Sample with the gage-eval Sample schema.
        """
        sample: Dict[str, Any] = dict(record)
        prompt = sample['messages'][0]['content'][0]['text']
        ans = sample['choices'][0]['message']['content'][0]['text']
        sample_id = str(hash(prompt))
        ref = [ans]
        final_messages = normalize_messages(sample["messages"])
        ret_sample = Sample(
            id = sample_id,
            schema_version = schema_version,
            messages = final_messages,
            references = ref
        )
        
        return ret_sample

if __name__ == '__main__':
    pass
