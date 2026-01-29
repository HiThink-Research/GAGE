"""ARC-AGI-2 preprocessors built on BasePreprocessor."""

from __future__ import annotations

from typing import Any, Dict
import json

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.sample import (
    SCHEMA_VERSION,
    Sample,
    Message,
    MessageContent,
)

from gage_eval.assets.datasets.preprocessors.arcagi2.utils import (
    create_prompt,
    extract_arcagi2_fields,
    format_grid,
)

from loguru import logger


class ARCAGI2Preprocessor(BasePreprocessor):
    """Preprocess ARC-AGI-2 dataset records into a standardized Sample.

    ARC-AGI-2 dataset fields:
    - train: List of input/output grid pairs for few-shot learning
    - test: List of test input grids (typically one, with expected output)
    - id/task_id: Unique identifier for the problem

    The dataset is a visual abstraction and reasoning benchmark where:
    - Each problem is a JSON file with train/test examples
    - Grids are 2D arrays of integers (0-9 representing different colors)
    - Task: discover the pattern from train examples and apply to test input
    """

    def to_sample(
        self,
        record: Dict[str, Any],
        *,
        schema_version: str = SCHEMA_VERSION,
        system_prompt: str | None = None,
        instruction: str | None = None,
        **kwargs: Any,
    ) -> Sample:
        """Converts a raw ARC-AGI-2 record into a standardized Sample.

        Args:
            record: Raw dataset record (dict with train/test grids).
            schema_version: Schema version for Sample.
            system_prompt: Optional system prompt to prepend.
            instruction: Optional instruction text to append.
            **kwargs: Reserved for forward compatibility.

        Returns:
            A Sample object compatible with the gage-eval Sample schema.
        """
        try:
            # STEP 1: Extract fields from the record
            fields = extract_arcagi2_fields(record)
            train_examples = fields["train_examples"]
            test_input = fields["test_input"]
            test_output = fields["test_output"]
            problem_id = fields["problem_id"]

            # STEP 2: Build metadata
            metadata: Dict[str, Any] = {
                "problem_id": problem_id,
                "train_example_count": len(train_examples),
                "train_examples": train_examples,  # Keep train examples in metadata
                "test_input": test_input,
            }

            # STEP 3: Create prompt
            prompt = create_prompt(
                record,
                system_prompt=system_prompt,
                instruction=instruction,
            )

            # STEP 4: Build messages
            message_content = MessageContent(type="text", text=prompt)
            messages: list[Message] = []

            if system_prompt:
                messages.append(
                    Message(
                        role="system",
                        content=[MessageContent(type="text", text=system_prompt.strip())],
                    )
                )

            messages.append(Message(role="user", content=[message_content]))

            # STEP 5: Build Sample
            # Use problem_id as sample_id for consistency
            sample_id = problem_id or str(hash(json.dumps(test_input, default=str)))

            # Use test_output as reference (expected answer)
            reference_output = format_grid(test_output) if test_output else ""

            ret_sample = Sample(
                id=sample_id,
                schema_version=schema_version,
                messages=messages,
                references=[test_output] if test_output else [],
                label=reference_output,
                metadata=metadata,
            )

            return ret_sample

        except Exception as e:
            logger.error("arcagi2_preprocessor failed: {}", e)
            raise


__all__ = ["ARCAGI2Preprocessor"]