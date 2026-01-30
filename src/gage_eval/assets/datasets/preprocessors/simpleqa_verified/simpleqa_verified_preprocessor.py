"""SimpleQA Verified preprocessors built on BasePreprocessor."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.sample import (
    SCHEMA_VERSION,
    Sample,
    Message,
    MessageContent,
)

from gage_eval.assets.datasets.preprocessors.simpleqa_verified.utils import (
    create_prompt,
    extract_simpleqa_verified_fields,
)

from loguru import logger


class SimpleQAVerifiedPreprocessor(BasePreprocessor):
    """Preprocess SimpleQA Verified dataset records into a standardized Sample.
    
    SimpleQA Verified dataset fields:
    - problem: The question/problem statement
    - answer: The gold answer
    - topic: Topic category (e.g., Politics, Art, Geography)
    - answer_type: Answer type (Person, Number, Date, Place, Other)
    - multi_step: Whether the question requires multiple steps
    - requires_reasoning: Whether the question requires reasoning
    - urls: List of URLs supporting the answer
    - original_index: Original index in the SimpleQA benchmark
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
        """Converts a raw SimpleQA Verified record into a standardized Sample.
        
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
            fields = extract_simpleqa_verified_fields(record)
            problem = fields["problem"]
            answer = fields["answer"]
            topic = fields["topic"]
            answer_type = fields["answer_type"]
            multi_step = fields["multi_step"]
            requires_reasoning = fields["requires_reasoning"]
            urls = fields["urls"]
            original_index = fields["original_index"]
            
            # STEP 2: Build metadata
            metadata: Dict[str, Any] = {
                "topic": topic,
                "answer_type": answer_type,
                "multi_step": multi_step,
                "requires_reasoning": requires_reasoning,
                "urls": urls,
                "original_index": original_index,
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
            # Generate sample_id: use hash for consistency with other preprocessors
            if original_index is not None:
                # Use hash of original_index for consistent numeric format
                sample_id = str(hash(str(original_index)))
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
            logger.error("simpleqa_verified_preprocessor failed: {}", e)
            raise


__all__ = ["SimpleQAVerifiedPreprocessor"]
