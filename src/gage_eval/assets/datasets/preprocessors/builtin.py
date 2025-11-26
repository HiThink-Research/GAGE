"""Builtin dataset preprocessors exposed via the registry."""

from __future__ import annotations

from gage_eval.assets.datasets.preprocessors.base import DatasetPreprocessor
from gage_eval.assets.datasets.preprocessors.registry import resolve_preprocess
from gage_eval.registry import registry


class _ModulePreprocessor(DatasetPreprocessor):
    module_path = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.module_path:
            raise ValueError("module_path must be defined for module-backed preprocessors")
        self._handle = resolve_preprocess(self.module_path)

    def transform(self, sample, **kwargs):
        merged = dict(self.kwargs)
        merged.update(kwargs)
        return self._handle.apply(sample, **merged)


@registry.asset(
    "dataset_preprocessors",
    "chat_prompt",
    desc="通用对话模板预处理器",
    tags=("prompt", "chat"),
    module="gage_eval.assets.datasets.preprocessors.preprocess",
)
class ChatPromptPreprocessor(_ModulePreprocessor):
    module_path = "gage_eval.assets.datasets.preprocessors.preprocess"


@registry.asset(
    "dataset_preprocessors",
    "cosy2",
    desc="COSY2 推理模版预处理器",
    tags=("prompt",),
    module="gage_eval.assets.datasets.preprocessors.preprocess_cosy2",
)
class Cosy2Preprocessor(_ModulePreprocessor):
    module_path = "gage_eval.assets.datasets.preprocessors.preprocess_cosy2"


@registry.asset(
    "dataset_preprocessors",
    "no_think",
    desc="No-think prompt 预处理器",
    tags=("prompt",),
    module="gage_eval.assets.datasets.preprocessors.preprocess_no_think",
)
class NoThinkPreprocessor(_ModulePreprocessor):
    module_path = "gage_eval.assets.datasets.preprocessors.preprocess_no_think"


@registry.asset(
    "dataset_preprocessors",
    "ruler",
    desc="RULER long-form prompt 预处理器",
    tags=("prompt",),
    module="gage_eval.assets.datasets.preprocessors.preprocess_ruler",
)
class RulerPreprocessor(_ModulePreprocessor):
    module_path = "gage_eval.assets.datasets.preprocessors.preprocess_ruler"


@registry.asset(
    "dataset_preprocessors",
    "multi_choice_standardizer",
    desc="多选题场景的标准化预处理器",
    tags=("prompt", "multiple-choice"),
    module="gage_eval.assets.datasets.preprocessors.preprocess_multi_choice",
)
class MultiChoicePreprocessor(_ModulePreprocessor):
    module_path = "gage_eval.assets.datasets.preprocessors.preprocess_multi_choice"


@registry.asset(
    "dataset_preprocessors",
    "multi_choice_struct_only",
    desc="多选题结构化预处理（仅补充 choices/metadata，不拼接 Prompt）",
    tags=("multiple-choice", "struct_only"),
    module="gage_eval.assets.datasets.preprocessors.preprocess_multi_choice_struct_only",
)
class MultiChoiceStructOnlyPreprocessor(_ModulePreprocessor):
    module_path = "gage_eval.assets.datasets.preprocessors.preprocess_multi_choice_struct_only"


@registry.asset(
    "dataset_preprocessors",
    "docvqa_image_standardizer",
    desc="DocVQA 文档问答多模态预处理器",
    tags=("prompt", "vision", "docvqa"),
    module="gage_eval.assets.datasets.preprocessors.preprocess_docvqa",
)
class DocVQAPreprocessor(_ModulePreprocessor):
    module_path = "gage_eval.assets.datasets.preprocessors.preprocess_docvqa"


@registry.asset(
    "dataset_preprocessors",
    "piqa_multi_choice",
    desc="PIQA 常识多选题提示词封装",
    tags=("prompt", "piqa", "multiple-choice"),
    module="gage_eval.assets.datasets.preprocessors.preprocess_piqa",
)
class PiqaPreprocessor(_ModulePreprocessor):
    module_path = "gage_eval.assets.datasets.preprocessors.preprocess_piqa"


@registry.asset(
    "dataset_preprocessors",
    "piqa_struct_only",
    desc="PIQA 常识多选题结构化预处理（仅补充 choices/metadata，不拼接 Prompt）",
    tags=("piqa", "multiple-choice", "struct_only"),
    module="gage_eval.assets.datasets.preprocessors.preprocess_piqa_struct_only",
)
class PiqaStructOnlyPreprocessor(_ModulePreprocessor):
    module_path = "gage_eval.assets.datasets.preprocessors.preprocess_piqa_struct_only"


@registry.asset(
    "dataset_preprocessors",
    "mmmu_multimodal_inputs",
    desc="MMMU 多模态 inputs 构造器（messages -> inputs.multi_modal_data）",
    tags=("prompt", "vision", "mmmu"),
    module="gage_eval.assets.datasets.preprocessors.preprocess_mmmu",
)
class MMMUMultimodalPreprocessor(_ModulePreprocessor):
    module_path = "gage_eval.assets.datasets.preprocessors.preprocess_mmmu"


@registry.asset(
    "dataset_preprocessors",
    "mmmu_multimodal_inputs",
    desc="MMMU 多模态任务的 inputs.multi_modal_data 预处理器",
    tags=("prompt", "vision", "mmmu"),
    module="gage_eval.assets.datasets.preprocessors.preprocess_mmmu",
)
class MMMUMultimodalPreprocessor(_ModulePreprocessor):
    module_path = "gage_eval.assets.datasets.preprocessors.preprocess_mmmu"


@registry.asset(
    "dataset_preprocessors",
    "mmmu_multimodal_inputs",
    desc="MMMU 多模态样本的 inputs.multi_modal_data 构造器",
    tags=("prompt", "vision", "mmmu"),
    module="gage_eval.assets.datasets.preprocessors.preprocess_mmmu",
)
class MMMUMultimodalPreprocessor(_ModulePreprocessor):
    module_path = "gage_eval.assets.datasets.preprocessors.preprocess_mmmu"


@registry.asset(
    "dataset_preprocessors",
    "mmmu_multimodal_inputs",
    desc="MMMU 多模态 inputs 预处理器（填充 inputs.multi_modal_data）",
    tags=("prompt", "vision", "mmmu"),
    module="gage_eval.assets.datasets.preprocessors.preprocess_mmmu",
)
class MMMUMultimodalPreprocessor(_ModulePreprocessor):
    module_path = "gage_eval.assets.datasets.preprocessors.preprocess_mmmu"
