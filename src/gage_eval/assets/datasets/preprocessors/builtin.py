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
from gage_eval.assets.datasets.preprocessors.appworld_preprocessor import AppWorldPreprocessor as NewAppWorld
from gage_eval.registry import registry

# benchmark GPQA-diamond
from gage_eval.assets.datasets.preprocessors.gpqa.gpqa_diamond_preprocessor import GpqaDiamondPreprocessor as NewGpqaDiamond

# benchmark MathVista
from gage_eval.assets.datasets.preprocessors.mathvista.mathvista_chat_preprocessor import MathVistaChatPreprocessor as NewMathVistaChat

# benchmark aime 2024
from gage_eval.assets.datasets.preprocessors.aime.aime2024 import AIME2024Preprocessor as NewAIME2024Preprocessor

# benchmark aime 2025
from gage_eval.assets.datasets.preprocessors.aime.aime2025 import AIME2025Preprocessor as NewAIME2025Preprocessor

# benchmark HLE (Humanity's Last Exam)
from gage_eval.assets.datasets.preprocessors.hle.hle_chat_converter import HLEConverter

# benchmark MMLU-Pro
from gage_eval.assets.datasets.preprocessors.mmlu_pro.mmlu_pro_converter import MMLUProConverter

# benchmark Math500
from gage_eval.assets.datasets.preprocessors.math500 import Math500Preprocessor

# benchmark MME
from gage_eval.assets.datasets.preprocessors.mme import MMEPreprocessor

# benchmark LiveCodeBench
from gage_eval.assets.datasets.preprocessors.live_code_bench.live_code_converter import LiveCodeBenchConverter

# benchmark global piqa
from gage_eval.assets.datasets.preprocessors.global_piqa.global_piqa_converter import GlobalPIQAConverter

# benchmark BizFinBench V2
from gage_eval.assets.datasets.preprocessors.biz_fin_bench_v2.biz_fin_bench_v2_converter import BizFinBenchV2Converter

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
    "appworld_preprocessor",
    desc="AppWorld JSONL preprocessor (task metadata + Sample envelope)",
    tags=("appworld", "agent"),
)
class AppWorldPreprocessor(NewAppWorld):
    """Standardize AppWorld JSONL records into the Sample schema."""

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


# benchmark GPQA-diamond
@registry.asset(
    "dataset_preprocessors",
    "gpqa_diamond_multi_choice",
    desc="GPQA diamond subset multiple-choice prompt wrapper",
    tags=("prompt", "gpqa", "gpqa_diamond", "multiple-choice"),
)
class GpqaDiamondPreprocessor(NewGpqaDiamond):
    pass

# benchmark MathVista
@registry.asset(
    "dataset_preprocessors",
    "mathvista_chat_preprocessor",
    desc="MathVista multimodal preprocessor (prompt + image + optional choices)",
    tags=("prompt", "vision", "mathvista"),
)
class MathVistaChatPreprocessor(NewMathVistaChat):
    pass

# benchmark aime2024
@registry.asset(
    "dataset_preprocessors",
    "aime2024_preprocessor",
    desc="AIME 2024 prompt wrapper",
    tags=("prompt", "aime2024"),
)
class AIME2024Preprocessor(NewAIME2024Preprocessor):
    pass


# benchmark aime2025
@registry.asset(
    "dataset_preprocessors",
    "aime2025_preprocessor",
    desc="AIME 2025 prompt wrapper",
    tags=("prompt", "aime2025"),
)
class AIME2025Preprocessor(NewAIME2025Preprocessor):
    pass

# benchmark HLE (Humanity's Last Exam)
@registry.asset(
    "dataset_preprocessors",
    "hle_preprocessor",
    desc="HLE prompt wrapper",
    tags=("prompt", "hle"),
)
class HLEPreprocessor(HLEConverter):
    pass

# benchmark MMLU-Pro
@registry.asset(
    "dataset_preprocessors",
    "mmlu_pro_chat_preprocessor",
    desc="MMLU Pro prompt wrapper",
    tags=("prompt", "mmlu-pro"),
)
class MMLUPreprocessor(MMLUProConverter):
    pass

# benchmark Math500
@registry.asset(
    "dataset_preprocessors",
    "math500",
    desc="MATH-500 dataset preprocessing logic",
    tags=("math", "math500", "latex"),
)
class Math500PreprocessorProvider(Math500Preprocessor):
    pass

# Also register with full name for backward compatibility
@registry.asset(
    "dataset_preprocessors",
    "math500_preprocessor",
    desc="MATH-500 dataset preprocessing logic (alias)",
    tags=("math", "math500", "latex"),
)
class Math500PreprocessorProviderAlias(Math500Preprocessor):
    pass

# benchmark MME
@registry.asset(
    "dataset_preprocessors",
    "mme",
    desc="MME dataset preprocessing logic",
    tags=("vision", "mme", "multi-modal"),
)
class MMEPreprocessorProvider(MMEPreprocessor):
    pass


@registry.asset(
    "dataset_preprocessors",
    "mme_preprocessor",
    desc="MME dataset preprocessing logic (alias)",
    tags=("vision", "mme", "multi-modal"),
)
class MMEPreprocessorProviderAlias(MMEPreprocessor):
    pass

# benchmark LiveCodeBench
@registry.asset(
    "dataset_preprocessors",
    "live_code_bench_chat_preprocessor",
    desc="Live Code Bench prompt wrapper",
    tags=("prompt", "live code bench"),
)
class LiveCodeBenchPreprocessor(LiveCodeBenchConverter):
    pass

# benchmark LiveCodeBench
@registry.asset(
    "dataset_preprocessors",
    "global_piqa_chat_preprocessor",
    desc="Global PIQA prompt wrapper",
    tags=("prompt", "global PIQA"),
)
class GlobalPIQAPreprocessor(GlobalPIQAConverter):
    pass

# benchmark BizFinBench V2
@registry.asset(
    "dataset_preprocessors",
    "bizfinbench_chat_preprocessor",
    desc="BizFinBench V2 prompt wrapper",
    tags=("prompt", "BizFinBench V2"),
)
class BizFinBenchV2Preprocessor(BizFinBenchV2Converter):
    pass