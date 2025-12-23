"""Builtin dataset preprocessors exposed via the registry."""

from __future__ import annotations

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.preprocessors.default_preprocessor import DefaultPreprocessor
from gage_eval.assets.datasets.preprocessors.multi_choice_preprocessor import MultiChoicePreprocessor as NewMultiChoice
from gage_eval.assets.datasets.preprocessors.docvqa_preprocessor import DocVQAPreprocessor as NewDocVQA
from gage_eval.assets.datasets.preprocessors.mathvista_preprocessor import (
    MathVistaPreprocessor as NewMathVista,
    MathVistaStructOnlyPreprocessor as NewMathVistaStructOnly,
)
from gage_eval.assets.datasets.preprocessors.mmmu_preprocessor import MMMUMultimodalPreprocessor as NewMMMU
from gage_eval.assets.datasets.preprocessors.piqa_preprocessor import (
    PiqaPreprocessor as NewPiqa,
    PiqaStructOnlyPreprocessor as NewPiqaStructOnly,
)
from gage_eval.assets.datasets.preprocessors.gpqa_preprocessor import (
    GpqaPreprocessor as NewGpqa,
    GpqaStructOnlyPreprocessor as NewGpqaStructOnly,
)
from gage_eval.registry import registry

# 1.benchmark GPQA-diamond
from gage_eval.assets.datasets.preprocessors.gpqa.gpqa_diamond_preprocessor import GpqaDiamondPreprocessor as NewGpqaDiamond


@registry.asset(
    "dataset_preprocessors",
    "multi_choice_standardizer",
    desc="多选题场景的标准化预处理器（新实现）",
    tags=("prompt", "multiple-choice"),
)
class MultiChoicePreprocessor(NewMultiChoice):
    pass


@registry.asset(
    "dataset_preprocessors",
    "docvqa_image_standardizer",
    desc="DocVQA 文档问答多模态预处理器（新实现）",
    tags=("prompt", "vision", "docvqa"),
)
class DocVQAPreprocessor(NewDocVQA):
    pass


@registry.asset(
    "dataset_preprocessors",
    "piqa_multi_choice",
    desc="PIQA 常识多选题提示词封装",
    tags=("prompt", "piqa", "multiple-choice"),
)
class PiqaPreprocessor(NewPiqa):
    pass


@registry.asset(
    "dataset_preprocessors",
    "piqa_struct_only",
    desc="PIQA 常识多选题结构化预处理（仅补充 choices/metadata，不拼接 Prompt）",
    tags=("piqa", "multiple-choice", "struct_only"),
)
class PiqaStructOnlyPreprocessor(NewPiqaStructOnly):
    pass


@registry.asset(
    "dataset_preprocessors",
    "mmmu_multimodal_inputs",
    desc="MMMU 多模态 inputs 构造器（messages -> inputs.multi_modal_data）",
    tags=("prompt", "vision", "mmmu"),
)
class MMMUMultimodalPreprocessor(NewMMMU):
    pass


@registry.asset(
    "dataset_preprocessors",
    "gpqa_multi_choice",
    desc="GPQA 常识多选题提示词封装",
    tags=("prompt", "gpqa", "multiple-choice"),
)
class GpqaPreprocessor(NewGpqa):
    pass

@registry.asset(
    "dataset_preprocessors",
    "gpqa_struct_only",
    desc="GPQA 常识多选题结构化预处理（仅补充 choices/metadata，不拼接 Prompt）",
    tags=("gpqa", "multiple-choice", "struct_only"),
)
class GpqaStructOnlyPreprocessor(NewGpqaStructOnly):
    pass


@registry.asset(
    "dataset_preprocessors",
    "mathvista_preprocess",
    desc="MathVista 多模态预处理（题干+图片+可选多选项）",
    tags=("prompt", "vision", "mathvista"),
)
class MathVistaPreprocessor(NewMathVista):
    pass


@registry.asset(
    "dataset_preprocessors",
    "mathvista_struct_only",
    desc="MathVista 多模态结构化预处理（仅补充多模态/choices/metadata，不拼接 Prompt）",
    tags=("mathvista", "vision", "struct_only"),
)
class MathVistaStructOnlyPreprocessor(NewMathVistaStructOnly):
    pass


# 1.benchmark GPQA-diamond
@registry.asset(
    "dataset_preprocessors",
    "gpqa_diamond_multi_choice",
    desc="GPQA diamond 子集多选题提示词封装",
    tags=("prompt", "gpqa", "gpqa_diamond", "multiple-choice"),
)
class GpqaDiamondPreprocessor(NewGpqaDiamond):
    pass