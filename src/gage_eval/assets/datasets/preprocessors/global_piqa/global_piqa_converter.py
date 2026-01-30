"""Global PIQA preprocessors."""

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
from gage_eval.assets.datasets.bundles.mmlu_pro.utils import format_cot_example, pretty_sample

_PROMPT = """Given the following situation, which option is more likely to be
correct?
Situation:
{question}

Option A: {opt_a}

Option B: {opt_b}

Your response should end with "The best answer is:
[answer_letter]" where [answer_letter] is one of A or B."""

class GlobalPIQAConverter(BasePreprocessor):
    """Preprocesses Global PIQA records into the Sample schema."""
    def to_sample(self, record: Dict[str, Any],
                  schema_version = SCHEMA_VERSION,
                  **kwargs: Any) -> Sample:
        """Converts a raw Global PIQA record into a standardized Sample.

        Args:
            record: Raw dataset record (typically a dict emitted by the loader).
            **kwargs: Reserved for forward compatibility.

        Returns:
            A Sample with the gage-eval Sample schema.
        """
        sample: Dict[str, Any] = dict(record)
        
        sample_id = str(sample.get("example_id"))        
        question  = sample.get("prompt")
        sol_1 = sample.get("solution0")
        sol_2 = sample.get("solution1")
        options = sample.get("options")
        answer_index = sample.get("label")
        language = sample.get("language")
        index_to_char = {
            0: 'A',
            1: 'B'
        }
        answer = index_to_char[answer_index]
        references = [answer]
        label = answer
        metadata = {
            "question": question,
            "options":  [sol_1, sol_2],
            "answer_index": answer_index,
        }
        prompt = _PROMPT.format(
            question=question,
            opt_a=sol_1,
            opt_b=sol_2)
        message_content = MessageContent(type="text", text=prompt)
        message = Message(role='user', content=[message_content])
            
        ret_sample = Sample(
            id = sample_id,
            schema_version = schema_version,
            messages = [message],
            metadata = metadata,
            references = references,
            label=label
        )
        return ret_sample

if __name__ == '__main__':
    sample = {
        "example_id": "0",
        "prompt": """answer question: """,
        "solution0": "what",
        "solution1": "why",
        "label": 1,
        "language": "english",
        "answer_index": 0,
    }
    t = GlobalPIQAConverter()
    ret = t.to_sample(sample)
    print("ret:", ret)
    print("len", len(ret.messages))
