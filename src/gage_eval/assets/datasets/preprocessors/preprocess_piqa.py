"""Preprocessor for the PIQA dataset."""

from __future__ import annotations

from typing import Any, Dict, List

from gage_eval.assets.datasets.preprocessors.preprocess_multi_choice import (
    convert_sample_to_inputs as _convert_multi_choice,
)


def convert_sample_to_inputs(
    sample: Dict[str, Any],
    *,
    system_prompt: str = (
        "你是一位擅长常识推理的助手，请阅读题目并在最后只输出正确选项对应的大写字母（A 或 B）。"
    ),
    instruction: str = "请判断哪个答案更合理，并在最后一行仅输出一个大写字母（A 或 B）。",
) -> List[Dict[str, Any]]:
    """Normalize PIQA samples into the standard multi-choice envelope."""

    sol1 = sample.get("sol1")
    sol2 = sample.get("sol2")
    if sol1 is None or sol2 is None:
        raise ValueError("PIQA sample must contain 'sol1' and 'sol2'")

    sample["choices"] = [str(sol1).strip(), str(sol2).strip()]
    return _convert_multi_choice(
        sample,
        question_field="goal",
        choices_field="choices",
        answer_field="label",
        answer_index_base=0,
        system_prompt=system_prompt,
        instruction=instruction,
    )
