"""HLE(Humanity's Last Exam) preprocessors."""

from __future__ import annotations

import random
from typing import Any, Dict


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


_SYSTEM_PROMPT = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"


class HLEConverter(BasePreprocessor):
    """Preprocesses HLE records into the Sample schema."""

    
    def to_sample(self, record: Dict[str, Any], schema_version = SCHEMA_VERSION, **kwargs: Any) -> Sample:
        """Converts a raw HLE record into a standardized Sample.

        Args:
            record: Raw dataset record (typically a dict emitted by the loader).
            **kwargs: Reserved for forward compatibility.

        Returns:
            A Sample with the gage-eval Sample schema.
        """
        sample: Dict[str, Any] = dict(record)
        
        sample_id = sample.get("id")

        question = sample.get('question')

        image_url = sample.get('image')

        answer = sample.get('answer')

        answer_type = sample.get('answer_type')

        author_name = sample.get('author_name')

        rationale = sample.get('rationale')

        raw_subject = sample.get('raw_subject')

        category = sample.get('category')

        canary = sample.get('canary')

        references = [answer]
        label = answer
        metadata = {
            "answer_type": answer_type,
            "author_name": author_name,
            "rationale": rationale,
            "raw_subject": raw_subject,
            "category": category,
            "canary": canary
        }

        
        sys_message_content = MessageContent(type="text", text=_SYSTEM_PROMPT)
        sys_message = Message(role='system', content = [sys_message_content])  
        text_message_content = MessageContent(type="text", text=question)        
        if image_url:
            image_message_content = MessageContent(type="image_url", image_url=dict(url=image_url))
            user_message = Message(role='user', content=[text_message_content, image_message_content])
        else:
            user_message = Message(role='user', content=[text_message_content])
            
        ret_sample = Sample(
            id = sample_id,
            schema_version = schema_version,
            messages = [sys_message, user_message],
            metadata = metadata,
            references = references,
            label=label
        )
        #print("ret_sample", ret_sample)
        #if image_url:
        #    exit(0)
        return ret_sample

if __name__ == '__main__':
        id = '6687ffb1091058ff19128813'
        question = "black to move"
        image = "data:image/jpeg;base64,/9j/4AAAAAA"
        answer  = "Rxf3, Rf1#"
        answer_type = "exactMatch"
        author_name = "Elliott T"
        rationale = "Chess engine says that this is the only mate i"
        raw_subject = "Chess"
        category = "Other"
        canary = "BENCHMARK DATA"
        sample = dict(
          id = id,
          image = image,
          answer = answer,
          answer_type = answer_type, 
          author_name = author_name,
          rationale = rationale,
          raw_subject = raw_subject,
          category = category,
          canary = canary 
        )

        pre = HLEConverter()
        
        ret = pre.to_sample(sample)
        print("ret: ", ret.messages)