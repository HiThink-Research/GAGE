"""GPQA preprocessors built on MultiChoicePreprocessor."""

from __future__ import annotations

import random
from typing import Any, Dict

from gage_eval.assets.datasets.preprocessors.multi_choice_preprocessor import MultiChoicePreprocessor
from gage_eval.assets.datasets.utils.rendering import strip_render_flags

from gage_eval.assets.datasets.preprocessors.gpqa.utils import (
    create_prompts,
    Example,
    load_examples,
)

from loguru import logger

from gage_eval.assets.datasets.utils.mapping import (
    extract_field,
    normalize_options,
    resolve_correct_choice,
)
from gage_eval.assets.datasets.utils.rendering import set_render_flags

_CHOICE_ALPHABET: Tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

class GpqaDiamondPreprocessor(MultiChoicePreprocessor):
    """GPQA 多选题预处理器。                                                                     
                                                                                                 
    处理逻辑：                                                                                   
    1. 提取 Question, Correct Answer 及 Incorrect Answer 1-3。                                   
    2. 合并并随机打乱选项。                                                                      
    3. 确定正确答案的索引。                                                                      
    4. 传递给基类进行标准 prompt 渲染。                                                          
    """

    def to_sample(self, record: Dict[str, Any], gpqa_prompt_type, **kwargs: Any) -> Dict[str, Any]:
        try:
            sample = dict(record)
            correct_answer = record.get("Correct Answer")
            examples = load_examples([sample])
            prompts, examples = create_prompts(examples, gpqa_prompt_type)
            assert len(prompts) == 1, "len(prompts) != 1" 
            assert len(examples) == 1, "len(examples) != 1"
            user_prompt =  prompts[0]
            example = examples[0]
            options = [example.choice1, example.choice2, example.choice3, example.choice4]
            answer_index = options.index(correct_answer)
            sample["question"] = user_prompt
            sample["choices"] = options
            # 整数索引传入
            sample["answer"] = answer_index
            options = normalize_options(options)
            if len(options) < 2:
                raise ValueError("Multiple-choice sample must provide at least two options")
            option_pairs = [(_CHOICE_ALPHABET[idx], text) for idx, text in enumerate(options)]
            if len(option_pairs) > len(_CHOICE_ALPHABET):
                raise ValueError("Multiple-choice preprocessor supports up to 26 options")                            
            correct_choice = resolve_correct_choice(answer_index, option_pairs, answer_index_base=0)
            if correct_choice is None:
                raise ValueError("Unable to resolve correct choice from answer field")                    
            messages: List[Dict[str, Any]] = []            
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                }   
            )
            sample["messages"] = messages
            sample["prompt"] = user_prompt
            set_render_flags(sample, mode="preprocess", source="manual", rendered_by="preprocess", cache_suffix="-converted")
            sample["choices"] = [
                {
                    "index": idx,
                    "label": label,
                    "message": {"role": "assistant", "content": [{"type": "text", "text": option_text}]},
                }
                for idx, (label, option_text) in enumerate(option_pairs)
            ]
            metadata = dict(sample.get("metadata") or {})
            metadata.update(
                {
                    "option_map": {label: option_text for label, option_text in option_pairs},
                    "correct_choice": correct_choice,
                    "question_field": "question",
                    "choices_field": "choices",
                    "answer_field": "answer",
                }
            )
            sample["metadata"] = metadata
            sample["inputs"] = sample.get("inputs") or {"prompt": user_prompt}
        except Exception as e:
            logger.warning("gpqa_diamond_preprocessor failed: {}", e)
        return sample
        





__all__ = ["GpqaDiamondPreprocessor"]