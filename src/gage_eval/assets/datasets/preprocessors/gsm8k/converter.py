"""GSM8K preprocessors."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict

from gage_eval.assets.datasets.sample import (
    SCHEMA_VERSION,
    Sample,
    Message,
    MessageContent,
)
from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor

_USER_PROMPT_TEMPLATE = """
Solve the following math problem step by step.
The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()

_ANS_SPLIT_RE = re.compile(r"\n####\s*")


class GSM8KPreprocessor(BasePreprocessor):
    """Preprocesses GSM8K records into the Sample schema."""

    def to_sample(
        self, record: Dict[str, Any], schema_version=SCHEMA_VERSION, **kwargs: Any
    ) -> Sample:
        """Converts a raw GSM8K record into a standardized Sample.

        Args:
            record: Raw dataset record (typically a dict emitted by the loader).
            **kwargs: Reserved for forward compatibility.

        Returns:
            A Sample with the gage-eval Sample schema.
        """
        sample: Dict[str, Any] = dict(record)

        question = sample.get("question", "")
        sample_id = hashlib.md5(question.encode("utf-8")).hexdigest()[:16]

        prompt = _USER_PROMPT_TEMPLATE.format(prompt=question)

        raw_answer = sample.get("answer", "")
        parts = _ANS_SPLIT_RE.split(raw_answer)
        if len(parts) >= 2:
            cot = parts[0]
            reference = parts[-1].strip()
        else:
            cot = raw_answer
            reference = ""

        references = [reference]
        label = reference
        metadata = {"cot": cot}

        message_content = MessageContent(type="text", text=prompt)
        message = Message(role="user", content=[message_content])

        ret_sample = Sample(
            id=sample_id,
            schema_version=schema_version,
            messages=[message],
            metadata=metadata,
            references=references,
            label=label,
        )
        return ret_sample
