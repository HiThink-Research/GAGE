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
from gage_eval.assets.datasets.preprocessors.grid_game_preprocessor import (
    GridGamePreprocessor as NewGridGame,
)
from gage_eval.assets.datasets.preprocessors.card_game_preprocessor import (
    CardGamePreprocessor as NewCardGame,
)
from gage_eval.assets.datasets.preprocessors.mmmu_preprocessor import MMMUMultimodalPreprocessor as NewMMMU
from gage_eval.assets.datasets.preprocessors.piqa_preprocessor import (
    PiqaPreprocessor as NewPiqa,
    PiqaStructOnlyPreprocessor as NewPiqaStructOnly,
)
from gage_eval.assets.datasets.preprocessors.swebench_pro_preprocessor import SwebenchProPreprocessor as NewSwebenchPro
from gage_eval.assets.datasets.preprocessors.gpqa_preprocessor import (
    GpqaPreprocessor as NewGpqa,
    GpqaStructOnlyPreprocessor as NewGpqaStructOnly,
)
from gage_eval.registry import registry

# 1.benchmark GPQA-diamond
from gage_eval.assets.datasets.preprocessors.gpqa.gpqa_diamond_preprocessor import GpqaDiamondPreprocessor as NewGpqaDiamond

# 2.benchmark MathVista
from gage_eval.assets.datasets.preprocessors.mathvista.mathvista_chat_preprocessor import MathVistaChatPreprocessor as NewMathVistaChat


@registry.asset(
    "dataset_preprocessors",
    "multi_choice_standardizer",
    desc="Multiple-choice standardizer preprocessor (new)",
    tags=("prompt", "multiple-choice"),
)
class MultiChoicePreprocessor(NewMultiChoice):
    pass


@registry.asset(
    "dataset_preprocessors",
    "docvqa_image_standardizer",
    desc="DocVQA multimodal preprocessor (new)",
    tags=("prompt", "vision", "docvqa"),
)
class DocVQAPreprocessor(NewDocVQA):
    pass


@registry.asset(
    "dataset_preprocessors",
    "grid_game_preprocessor",
    desc="Grid game preprocessor (board metadata + sample envelope)",
    tags=("grid", "game"),
)
class GridGamePreprocessor(NewGridGame):
    """Standardize grid game records into the Sample schema."""

    pass


@registry.asset(
    "dataset_preprocessors",
    "card_game_preprocessor",
    desc="Card game preprocessor (player metadata + sample envelope)",
    tags=("card", "game"),
)
class CardGamePreprocessor(NewCardGame):
    """Standardize card game records into the Sample schema."""

    pass


@registry.asset(
    "dataset_preprocessors",
    "piqa_multi_choice",
    desc="PIQA multiple-choice prompt wrapper",
    tags=("prompt", "piqa", "multiple-choice"),
)
class PiqaPreprocessor(NewPiqa):
    pass


@registry.asset(
    "dataset_preprocessors",
    "piqa_struct_only",
    desc="PIQA structured preprocessor (choices/metadata only; no prompt concatenation)",
    tags=("piqa", "multiple-choice", "struct_only"),
)
class PiqaStructOnlyPreprocessor(NewPiqaStructOnly):
    pass


@registry.asset(
    "dataset_preprocessors",
    "gpqa_multi_choice",
    desc="GPQA multiple-choice prompt wrapper",
    tags=("prompt", "gpqa", "multiple-choice"),
)
class GpqaPreprocessor(NewGpqa):
    pass


@registry.asset(
    "dataset_preprocessors",
    "gpqa_struct_only",
    desc="GPQA structured preprocessor (choices/metadata only; no prompt concatenation)",
    tags=("gpqa", "multiple-choice", "struct_only"),
)
class GpqaStructOnlyPreprocessor(NewGpqaStructOnly):
    pass


@registry.asset(
    "dataset_preprocessors",
    "mmmu_multimodal_inputs",
    desc="MMMU multimodal inputs builder (messages -> inputs.multi_modal_data)",
    tags=("prompt", "vision", "mmmu"),
)
class MMMUMultimodalPreprocessor(NewMMMU):
    pass


@registry.asset(
    "dataset_preprocessors",
    "gpqa_multi_choice",
    desc="GPQA commonsense multiple-choice prompt wrapper",
    tags=("prompt", "gpqa", "multiple-choice"),
)
class GpqaPreprocessor(NewGpqa):
    pass

@registry.asset(
    "dataset_preprocessors",
    "gpqa_struct_only",
    desc="GPQA commonsense structured preprocessor (choices/metadata only; no prompt concatenation)",
    tags=("gpqa", "multiple-choice", "struct_only"),
)
class GpqaStructOnlyPreprocessor(NewGpqaStructOnly):
    pass


@registry.asset(
    "dataset_preprocessors",
    "mathvista_preprocessor",
    desc="MathVista multimodal preprocessor (prompt + image + optional choices)",
    tags=("prompt", "vision", "mathvista"),
)
class MathVistaPreprocessor(NewMathVista):
    pass

@registry.asset(
    "dataset_preprocessors",
    "mathvista_struct_only",
    desc="MathVista structured multimodal preprocessor (multimodal/choices/metadata only; no prompt concatenation)",
    tags=("mathvista", "vision", "struct_only"),
)
class MathVistaStructOnlyPreprocessor(NewMathVistaStructOnly):
    pass


# 1.benchmark GPQA-diamond
@registry.asset(
    "dataset_preprocessors",
    "gpqa_diamond_multi_choice",
    desc="GPQA diamond subset multiple-choice prompt wrapper",
    tags=("prompt", "gpqa", "gpqa_diamond", "multiple-choice"),
)
class GpqaDiamondPreprocessor(NewGpqaDiamond):
    pass

# 2.benchmark MathVista
@registry.asset(
    "dataset_preprocessors",
    "mathvista_chat_preprocessor",
    desc="MathVista multimodal preprocessor (prompt + image + optional choices)",
    tags=("prompt", "vision", "mathvista"),
)
class MathVistaChatPreprocessor(NewMathVistaChat):
    pass
