"""MMSU preprocessors."""

from __future__ import annotations
import os
import random
from typing import Any, Dict
import transformers
import base64

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

def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

def build_audio_file_path(root, audio_path):
    if root is None or audio_path is None:
        raise ValueError(f"root or audio_path value is None")
    ret = os.path.join(root,audio_path.strip('/'))
    return ret

def build_answer_index(choice_list, answer):
    i_to_char = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D'
    }
    for i, choice in enumerate(choice_list):
        if choice == answer:
            return i_to_char[i]
    return 'E'

_TEXT_PROMPT = """
Choose the most suitable answer from options A, B, C, and D to respond the question in next line, **you should only choose A or B or C or D.** Do not provide any additional explanations or content.
Question: {question}
A. {choice_a}, B. {choice_b}, C. {choice_c}, D. {choice_d}.
"""

class MMSUConverter(BasePreprocessor):
    """Preprocesses MMSU records into the Sample schema."""
    def to_sample(self, record: Dict[str, Any],
                  schema_version = SCHEMA_VERSION,
                  audio_path_root = None,
                  **kwargs: Any) -> Sample:
        """Converts a raw Global MMSU record into a standardized Sample.

        Args:
            record: Raw dataset record (typically a dict emitted by the loader).
            **kwargs: Reserved for forward compatibility.

        Returns:
            A Sample with the gage-eval Sample schema.
        """
        try:
            sample: Dict[str, Any] = dict(record)
            sample_id = sample.get("id")
            task_name = sample.get("task_name")
            question = sample.get("question")
            choice_a = sample.get("choice_a")
            choice_b = sample.get("choice_b")
            choice_c = sample.get("choice_c")
            choice_d = sample.get("choice_d")
            answer = sample.get("answer_gt")
            category = sample.get("category")
            sub_category = sample.get("sub-category")
            sub_sub_category = sample.get("sub-sub-category")
            linguistics_sub_discipline = sample.get("linguistics_sub_discipline")
            audio_path = sample.get("audio_path")
            audio_file_path = build_audio_file_path(audio_path_root, audio_path)
            audio_base64 = encode_audio(audio_file_path)
            audio_frag = MessageContent(**{"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}})            
            label = build_answer_index(
                [choice_a, choice_b, choice_c, choice_d],
                answer
            )
            references = [label]
            metadata = {
                "task_name": task_name,
                "category": category,
                "sub_category": sub_category,
                "sub_sub_category": sub_sub_category,
                "linguistics_sub_discipline": linguistics_sub_discipline
            }
            prompt = _TEXT_PROMPT.format(
                question=question,
                choice_a=choice_a,
                choice_b=choice_b,
                choice_c=choice_c,
                choice_d=choice_d
            )
            content = [audio_frag, 
                MessageContent(**{"type": "text", "text": prompt})
            ]
            message = Message(**{"role": "user", "content": content})
            ret_sample = Sample(
                id = sample_id,
                schema_version = schema_version,
                messages = [message],
                metadata = metadata,
                references = references,
                label=label
            )
        except Exception as e:
            logger.error(f"val error {e}")
        #print("sample:", ret_sample)
        return ret_sample
        # exit(0)

if __name__ == '__main__':
    sample = {"id": "code_switch_question_answering_7affd9a4-03cf-4eee-a3be-e4580a8df238", "task_name": "code_switch_question_answering", "audio_path": "/audio/code_switch_question_answering_7affd9a4-03cf-4eee-a3be-e4580a8df238.wav", "question": "What did the speaker receive from their friend?", "choice_a": "A cake.", "choice_b": "Chocolate.", "choice_c": "A fruit basket.", "choice_d": "A drink.", "answer_gt": "Chocolate.", "category": "Reasoning", "sub-category": "Linguistics", "sub-sub-category": "Semantics", "linguistics_sub_discipline": "Semantics"}
    pre = MMSUConverter()
    ret = pre.to_sample(sample,
                            audio_path_root="/mnt/aime_data_ssd/user_workspace/zhuwenqiao/GAGE_workspace/data/mmsu")
    print("ret:", ret)
    print("len", len(ret.messages))
