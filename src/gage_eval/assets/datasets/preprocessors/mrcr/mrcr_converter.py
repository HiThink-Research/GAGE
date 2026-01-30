"""BizFinBench v2 preprocessors."""

from __future__ import annotations

import random
from typing import Any, Dict
import transformers
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

from gage_eval.assets.datasets.preprocessors.live_code_bench.utils import normalize_messages

def mock_sample():
    return {
        "prompt":  '[{"role": "user", "content": "are you ok?"}]',
        "answer": "mWEa9DrPT3**Verse 1** \nIn a world so vast",
        "random_string_to_prepend": "mWEa9DrPT3",
        "n_needles": 2,
        "desired_msg_index": 721,
        "total_messages": 772,
        "n_chars": 708925,
        "date_added": "04-12-2025"
    }

class MRCRConverter(BasePreprocessor):
    """Preprocesses MRCR records into the Sample schema."""
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
        # sample = mock_sample()
        try:
             prompt = sample.get('prompt')
             messages =  json.loads(prompt)
             ans = sample.get('answer')
             random_string_to_prepend = sample.get('random_string_to_prepend')
             n_needles = sample.get('n_needles')
             desired_msg_index = sample.get('desired_msg_index')
             date_added = sample.get('date_added')
        except Exception as e:
            print("e:", e)
            prompt = ""
            messages = []
            ans = ""
            random_string_to_prepend = ""
            n_needles = 0
            desired_msg_index = 0
            date_added = 0
        sample_id = str(hash(prompt))
        ref = [ans]
        metadata = {
            "random_string_to_prepend":random_string_to_prepend,
            "n_needles": n_needles,
            "desired_msg_index": desired_msg_index,
            "date_added": date_added
        }
        #message_content = MessageContent(type="text", text=prompt)
        #messages = Message(role='user', content=[message_content])
        final_messages = normalize_messages(messages)    
        ret_sample = Sample(
            id = sample_id,
            schema_version = schema_version,
            messages = final_messages,
            references = ref,
            metadata = metadata
        )
        return ret_sample

if __name__ == '__main__':
    pre = MRCRConverter()
    ret = pre.to_sample({})
    print("ret", ret)