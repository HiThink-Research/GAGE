"""PIQA preprocessors built on MultiChoicePreprocessor."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.assets.datasets.preprocessors.multi_choice_preprocessor import MultiChoicePreprocessor


class PiqaPreprocessor(MultiChoicePreprocessor):
    """PIQA 多选题提示词封装，可通过 kwargs 调整提示。"""

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sample = dict(record)
        # 兼容原始 PIQA 字段（goal/sol1/sol2/label）与已预结构化的 question/choices/answer
        question = record.get("question") or record.get("goal")
        options = record.get("choices")
        if not options:
            options = [record.get("sol1"), record.get("sol2")]
        answer = record.get("answer", record.get("label"))

        sample["question"] = question
        sample["choices"] = options
        sample["answer"] = answer
        return super().to_sample(
            sample,
            question_field="question",
            options_field="choices",
            answer_field="answer",
            answer_index_base=0,
            **kwargs,
        )


class PiqaStructOnlyPreprocessor(PiqaPreprocessor):
    """PIQA 结构化预处理（不渲染 prompt，仅保留 choices/metadata）。"""

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sample = super().to_sample(record, **kwargs)
        # 保留 messages/choices 以满足 Envelope 校验，仅移除渲染标记与 prompt 输入
        sample.pop("prompt", None)
        sample["messages"] = []
        sample["inputs"] = {}
        sample.pop("chat_template_mode", None)
        sample.pop("rendered_by", None)
        sample.pop("template_source", None)
        sample.pop("cache_suffix", None)
        return sample


__all__ = ["PiqaPreprocessor", "PiqaStructOnlyPreprocessor"]
from gage_eval.assets.datasets.utils.rendering import strip_render_flags
