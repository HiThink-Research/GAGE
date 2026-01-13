"""MATH-500 preprocessors built on BasePreprocessor."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.sample import (
    SCHEMA_VERSION,
    Sample,
    Message,
    MessageContent,
)

from gage_eval.assets.datasets.preprocessors.math500.utils import (
    create_prompt,
    extract_math500_fields,
)

from loguru import logger


class Math500Preprocessor(BasePreprocessor):
    """Preprocess MATH-500 dataset records into a standardized Sample.
    
    MATH-500 dataset fields:
    - problem: The problem statement (LaTeX format)
    - solution: The solution steps (LaTeX format)
    - answer: The final answer (LaTeX format)
    - subject: Subject category (e.g., Precalculus, Algebra)
    - level: Difficulty level (1-5)
    - unique_id: Unique identifier for the problem
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
        """Converts a raw MATH-500 record into a standardized Sample.
        
        Args:
            record: Raw dataset record (typically a dict emitted by the loader).
            schema_version: Schema version for Sample.
            system_prompt: Optional system prompt to prepend.
            instruction: Optional instruction text to append.
            **kwargs: Reserved for forward compatibility.
        
        Returns:
            A Sample object compatible with the gage-eval Sample schema.
        """
        try:
            # STEP 1: Extract fields from the record
            fields = extract_math500_fields(record)
            problem = fields["problem"]
            answer = fields["answer"]
            solution = fields["solution"]
            subject = fields["subject"]
            level = fields["level"]
            unique_id = fields["unique_id"]
            
            # STEP 2: Build metadata
            metadata: Dict[str, Any] = {
                "subject": subject,
                "level": level,
                "unique_id": unique_id,
                "solution": solution,  # Keep solution in metadata for reference
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
            # Generate sample_id: use hash for consistency with other preprocessors (gpqa, etc.)
            if unique_id:
                # Use hash of unique_id for consistent numeric format (like gpqa)
                sample_id = str(hash(unique_id))
            else:
                # Fallback to hash of problem text
                sample_id = str(hash(problem))
            
            ret_sample = Sample(
                id=sample_id,
                schema_version=schema_version,
                messages=messages,
                references=[answer] if answer else [],
                label=answer,
                metadata=metadata,
            )
            
            return ret_sample
            
        except Exception as e:
            logger.error("math500_preprocessor failed: {}", e)
            raise


__all__ = ["Math500Preprocessor"]

