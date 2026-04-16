"""AIME2026 preprocessors."""

from __future__ import annotations

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


class AIME2026Preprocessor(BasePreprocessor):
    """Preprocesses AIME2026 records into the Sample schema."""

    def to_sample(
        self, record: Dict[str, Any], schema_version=SCHEMA_VERSION, **kwargs: Any
    ) -> Sample:
        """Converts a raw AIME2026 record into a standardized Sample.

        Args:
            record: Raw dataset record (typically a dict emitted by the loader).
            **kwargs: Reserved for forward compatibility.

        Returns:
            A Sample with the gage-eval Sample schema.
        """
        sample: Dict[str, Any] = dict(record)

        sample_id = sample.get("id")

        prompt = _USER_PROMPT_TEMPLATE.format(prompt=sample.get("problem"))

        answer = sample.get("answer")

        references = [answer]
        label = answer

        message_content = MessageContent(type="text", text=prompt)
        message = Message(role="user", content=[message_content])

        ret_sample = Sample(
            id=sample_id,
            schema_version=schema_version,
            messages=[message],
            references=references,
            label=label,
        )
        return ret_sample
