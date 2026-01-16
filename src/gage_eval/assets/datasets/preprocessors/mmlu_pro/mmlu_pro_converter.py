"""MMLU-Pro preprocessors."""

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

_PROMPT_PREFIX = """The following are multiple choice questions (with answers) about {$}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice."""

def generate_cot_prompt(few_shots, curr, k):
    few_shots = few_shots[:k]
    prompt = _PROMPT_PREFIX
    subject = curr["category"]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example_str in few_shots:
        prompt += example_str
    prompt += format_cot_example(curr, including_answer=False)
    return prompt

def build_prompt(self, sample,
                     few_shot_examples,
                     n_few_shot=5,
                  tokenizer_path = None,
                  max_length = 4094,
                  max_new_tokens = 2048):
    if n_few_shot <= 0 or few_shot_examples is None:
        return generate_cot_prompt([], sample, 0)
    else:
        if tokenizer_path is None:
            few_shots = few_shot_examples[:n_few_shot]
            return generate_cot_prompt(few_shots, sample, n_few_shot)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True)
            prompt_length_ok = False
            prompt = None
            while not prompt_length_ok:
                few_shots = few_shot_examples[:n_few_shot]
                prompt = generate_cot_prompt(few_shots, curr, k)
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {key: value.cuda() for key, value in inputs.items()}
                length = len(inputs["input_ids"][0])
                if length < max_length - max_new_tokens:
                    prompt_length_ok = True
                k -= 1
            return prompt

class MMLUProConverter(BasePreprocessor):
    """Preprocesses MMLU-Pro records into the Sample schema."""
    def to_sample(self, record: Dict[str, Any],
                  schema_version = SCHEMA_VERSION,
                  n_few_shot=5,
                  tokenizer_path = None,
                  max_length = 4094,
                  max_new_tokens = 2048,
                  **kwargs: Any) -> Sample:
        """Converts a raw MMLU-Pro record into a standardized Sample.

        Args:
            record: Raw dataset record (typically a dict emitted by the loader).
            **kwargs: Reserved for forward compatibility.

        Returns:
            A Sample with the gage-eval Sample schema.
        """
        sample: Dict[str, Any] = pretty_sample(dict(record))
        
        sample_id = str(sample.get("question_id"))        
        question  = sample.get("question")
        options = sample.get("options")
        answer = sample.get("answer")
        answer_index = sample.get("answer_index")
        few_shot_examples = sample.get("few_shot_examples")
        cot_content = sample.get("cot_content")
        category = sample.get("category")
        src = sample.get("src")

        references = [answer]
        label = answer
        metadata = {
            "question": question,
            "options":  options,
            "answer_index": answer_index,
            "cot_content": cot_content,
            "category": category,
            "src": src
        }
        prompt = build_prompt(self, sample,
                     few_shot_examples,
                     n_few_shot=5,
                  tokenizer_path = None,
                  max_length = 4094,
                  max_new_tokens = 2048)
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
        "question_id": "0",
        "question": """The symmetric group $S_n$ has $
\factorial{n}$ elements, hence it is not true that $S_{10}$ has 10 elements.
Find the characteristic of the ring 2Z.""",
        "options": ["0","30","3","10","12","50","2","100","20","5"],
        "answer": 'A',
        "answer_index": 0,
        "cot_content": """	
A: Let's think step by step. A characteristic of a ring is R is $n$ if the statement $ka = 0$ for all $a\in 2Z$ implies that $k$ is a multiple of $n$. Assume that $ka = 0$ for all $a\in 2Z$ for some $k$. In particular $2k = 0$. Hence $k=0$ and $n=0$. The answer is (A).""",
        "category": "math",
        "src": "cot_lib-abstract_algebra"
    }
    t = MMLUProConverter()
    ret = t.to_sample(sample)
    print("ret:", ret)
    print("len", len(ret.messages))
