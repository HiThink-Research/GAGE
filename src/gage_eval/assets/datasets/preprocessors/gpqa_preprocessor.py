"""GPQA preprocessors built on MultiChoicePreprocessor."""

from __future__ import annotations

import random
from typing import Any, Dict

from gage_eval.assets.datasets.preprocessors.multi_choice_preprocessor import MultiChoicePreprocessor
from gage_eval.assets.datasets.utils.rendering import (
    contains_multimodal,
    render_messages_with_fallback,
    set_render_flags,
    strip_render_flags,
)


class GpqaPreprocessor(MultiChoicePreprocessor):
    """GPQA 多选题预处理器。                                                                     
                                                                                                 
    处理逻辑：                                                                                   
    1. 提取 Question, Correct Answer 及 Incorrect Answer 1-3。                                   
    2. 合并并随机打乱选项。                                                                      
    3. 确定正确答案的索引。                                                                      
    4. 传递给基类进行标准 prompt 渲染。                                                          
    """

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sample = dict(record)

        # step1: 提取字段
        question = record.get("Question")
        correct_answer = record.get("Correct Answer")
        incorrect_answers = [
            record.get("Incorrect Answer 1"),
            record.get("Incorrect Answer 2"),
            record.get("Incorrect Answer 3"),
        ]

        # 完整性检查
        if question is None or correct_answer is None:
            pass

        # step2: 构建选项列表
        options = [correct_answer] + [opt for opt in incorrect_answers if opt is not None]

        # step3: 随机打乱选项
        random.shuffle(options)

        # step4：获取正确选项索引
        try:
            answer_index = options.index(correct_answer)
        except ValueError:
            answer_index = 0

        # step5：更新sample字典，适配父类MultiChoicePreprocessor
        sample["question"] = question
        sample["choices"] = options
        # 整数索引传入
        sample["answer"] = answer_index

        # step6： 调用父类
        sample = super().to_sample(
            sample,
            question_field="question",
            options_field="choices",
            answer_field="answer",
            answer_index_base=0,
            **kwargs,
        )
        # 尝试复用 tokenizer 的 chat_template 渲染文本并生成 input_ids（对齐 llm-eval 行为）
        messages = sample.get("messages")
        if (
            self._tokenizer is not None
            and isinstance(messages, list)
            and messages
            and not contains_multimodal(messages)
        ):
            prompt, source = render_messages_with_fallback(messages, self._tokenizer)
            sample["prompt"] = prompt
            sample["inputs"] = {"prompt": prompt}
            try:
                if hasattr(self._tokenizer, "encode"):
                    sample["inputs"]["input_ids"] = self._tokenizer.encode(prompt)
            except Exception:
                # encoding 失败则只保留 prompt，避免中断预处理
                pass
            set_render_flags(
                sample,
                mode="preprocess",
                source=source,
                rendered_by="preprocess",
                cache_suffix="-chat_template" if source == "model" else "-plain",
            )
            if getattr(self, "_tokenizer_path", None) and "_tokenizer_path" not in sample:
                sample["_tokenizer_path"] = self._tokenizer_path
        return sample

class GpqaStructOnlyPreprocessor(GpqaPreprocessor):
    """GPQA 结构化预处理（不渲染 prompt，仅保留 choices/metadata）。"""

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sample = super().to_sample(record, **kwargs)
        # 保留 messages/choices 以满足 Envelope 校验，仅移除渲染标记与 prompt 输入
        sample.pop("prompt", None)
        sample["messages"] = []
        sample["inputs"] = {}
        strip_render_flags(sample)
        return sample

__all__ = ["GpqaPreprocessor", "GpqaStructOnlyPreprocessor"]
