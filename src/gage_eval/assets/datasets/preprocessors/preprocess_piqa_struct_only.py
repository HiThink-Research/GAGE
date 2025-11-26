"""PIQA 结构化预处理器：仅规范化多选结构，不拼接 Prompt。"""

from __future__ import annotations

from typing import Any, Dict, List

from gage_eval.assets.datasets.preprocessors.preprocess_piqa import (
    convert_sample_to_inputs as _convert_piqa_inputs,
)


def convert_sample_to_inputs(sample: Dict[str, Any], **kwargs: Any) -> List[Dict[str, Any]]:
    """Normalize PIQA samples while keeping prompt逻辑在外部 Adapter。

    行为说明：
    - 复用 `preprocess_piqa.convert_sample_to_inputs` 的结构化能力：
      - 填充 `choices` 列表；
      - 写入 `metadata.option_map` 与 `metadata.correct_choice` 等字段；
    - 随后清理 `messages` 与 `prompt` 字段，让后续 Prompt 由 ModelRoleAdapter 渲染；
    - 返回空列表，意味着 JSONL 记录上的 `inputs` 字段不携带自然语言 Prompt。
    """

    # 复用原始 PIQA 预处理逻辑进行结构化，但忽略外部 kwargs（例如 data_path），
    # 避免向下游函数传入未声明的参数。
    _convert_piqa_inputs(sample)
    # 清理 preprocessor 注入的自然语言 Prompt，由外部 PromptAsset 接管
    sample.pop("messages", None)
    sample.pop("prompt", None)
    return []
