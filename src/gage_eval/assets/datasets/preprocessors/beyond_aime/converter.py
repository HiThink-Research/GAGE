"""BeyondAIME preprocessor."""

from __future__ import annotations

import hashlib
from typing import Any, Dict

from gage_eval.assets.datasets.sample import (
    SCHEMA_VERSION,
    Sample,
    Message,
    MessageContent,
)

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor

_USER_PROMPT_TEMPLATE = """{problem}

Put your final answer within \\boxed{{}}."""


class BeyondAIMEPreprocessor(BasePreprocessor):
    """Preprocesses BeyondAIME records into the Sample schema."""

    def to_sample(
        self, record: Dict[str, Any], schema_version: str = SCHEMA_VERSION, **kwargs: Any
    ) -> Sample:
        """Converts a raw BeyondAIME record into a standardized Sample.

        Args:
            record: Raw dataset record (typically a dict emitted by the loader).
            schema_version: Schema version string.
            **kwargs: Reserved for forward compatibility.

        Returns:
            A Sample with the gage-eval Sample schema.
        """
        sample: Dict[str, Any] = dict(record)

        # Extract fields from record
        problem = sample.get("problem", "")
        answer = sample.get("answer", "")

        # Generate sample_id using hash of problem
        sample_id = hashlib.md5(problem.encode("utf-8")).hexdigest()

        # Format prompt with the required suffix
        prompt = _USER_PROMPT_TEMPLATE.format(problem=problem)

        # Build references and label
        references = [answer]
        label = answer

        # Build metadata
        metadata = {}

        # Build messages
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
