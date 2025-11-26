"""Multi-choice structural preprocessor: normalize choices/metadata without attaching prompts."""

from __future__ import annotations

from typing import Dict, List, Optional

from gage_eval.assets.datasets.preprocessors.preprocess_multi_choice import (
    convert_sample_to_inputs as _convert_multi_choice_inputs,
)


def convert_sample_to_inputs(
    sample: Dict[str, Any],
    *,
    question_field: str = "question",
    options_field: Optional[str] = None,
    choices_field: Optional[str] = None,
    answer_field: str = "answer",
    answer_index_base: int = 0,
    system_prompt: Optional[str] = None,
    instruction: Optional[str] = "请仅输出正确选项对应的大写字母，例如 'A'。",
) -> List[Dict[str, Any]]:
    """Normalize multiple-choice samples while keeping prompt logic in external adapters.

    行为说明：
    - 复用 `preprocess_multi_choice.convert_sample_to_inputs` 的结构化能力：
      - 归一化选项并填充 `choices` 列表；
      - 写入 `metadata.option_map` 与 `metadata.correct_choice` 等字段；
    - 随后清理 `messages` 与 `prompt` 字段，让后续 Prompt 由 ModelRoleAdapter 渲染；
    - 返回空列表，意味着 JSONL 记录上的 `inputs` 字段不携带自然语言 Prompt。
    """

    _convert_multi_choice_inputs(
        sample,
        question_field=question_field,
        options_field=options_field,
        choices_field=choices_field,
        answer_field=answer_field,
        answer_index_base=answer_index_base,
        system_prompt=system_prompt,
        instruction=instruction,
    )
    sample.pop("messages", None)
    sample.pop("prompt", None)
    return []
