"""Builtin dataset preprocessors exposed via the registry."""

from __future__ import annotations

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.preprocessors.default_preprocessor import DefaultPreprocessor
from gage_eval.assets.datasets.preprocessors.multi_choice_preprocessor import MultiChoicePreprocessor as NewMultiChoice
import warnings

from gage_eval.assets.datasets.preprocessors.docvqa_preprocessor import DocVQAPreprocessor as NewDocVQA
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
try:
    from gage_eval.assets.datasets.preprocessors.mathvista_preprocessor import (
        MathVistaPreprocessor as NewMathVista,
        MathVistaStructOnlyPreprocessor as NewMathVistaStructOnly,
    )
except Exception as exc:  # pragma: no cover - optional dependency guard
    NewMathVista = None
    NewMathVistaStructOnly = None
    warnings.warn(f"MathVista preprocessors unavailable: {exc}", RuntimeWarning)

try:
    from gage_eval.assets.datasets.preprocessors.mathvista.mathvista_chat_preprocessor import (
        MathVistaChatPreprocessor as NewMathVistaChat,
    )
except Exception as exc:  # pragma: no cover - optional dependency guard
    NewMathVistaChat = None
    warnings.warn(f"MathVista chat preprocessor unavailable: {exc}", RuntimeWarning)

# benchmark aime 2024
from gage_eval.assets.datasets.preprocessors.aime.aime2024 import AIME2024Preprocessor as NewAIME2024Preprocessor

# benchmark aime 2025
from gage_eval.assets.datasets.preprocessors.aime.aime2025 import AIME2025Preprocessor as NewAIME2025Preprocessor

# benchmark HLE (Humanity's Last Exam)
from gage_eval.assets.datasets.preprocessors.hle.hle_chat_converter import HLEConverter

# benchmark MMLU-Pro
try:
    from gage_eval.assets.datasets.preprocessors.mmlu_pro.mmlu_pro_converter import MMLUProConverter as NewMMLUProConverter
except Exception as exc:  # pragma: no cover - optional dependency guard
    NewMMLUProConverter = None
    warnings.warn(f"MMLU-Pro preprocessor unavailable: {exc}", RuntimeWarning)

# benchmark Math500
from gage_eval.assets.datasets.preprocessors.math500 import Math500Preprocessor

# benchmark MME
try:
    from gage_eval.assets.datasets.preprocessors.mme import MMEPreprocessor as NewMMEPreprocessor
except Exception as exc:  # pragma: no cover - optional dependency guard
    NewMMEPreprocessor = None
    warnings.warn(f"MME preprocessor unavailable: {exc}", RuntimeWarning)


# benchmark SimpleQA Verified
from gage_eval.assets.datasets.preprocessors.simpleqa_verified import SimpleQAVerifiedPreprocessor

# benchmark ARC-AGI-2
from gage_eval.assets.datasets.preprocessors.arcagi2 import ARCAGI2Preprocessor

# benchmark ScreenSpot-Pro
from gage_eval.assets.datasets.preprocessors.screespot_pro import ScreenSpotProPreprocessor
from gage_eval.assets.datasets.preprocessors.charxiv import CharXivReasoningPreprocessor

# benchmark LiveCodeBench
from gage_eval.assets.datasets.preprocessors.live_code_bench.live_code_converter import LiveCodeBenchConverter

# benchmark global piqa
from gage_eval.assets.datasets.preprocessors.global_piqa.global_piqa_converter import GlobalPIQAConverter

# benchmark BizFinBench V2
from gage_eval.assets.datasets.preprocessors.biz_fin_bench_v2.biz_fin_bench_v2_converter import BizFinBenchV2Converter

# benchmark MRCR
from gage_eval.assets.datasets.preprocessors.mrcr.mrcr_converter import MRCRConverter



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


if NewMathVista is not None:
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
if NewMathVistaChat is not None:
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
if NewMMLUProConverter is not None:
    @registry.asset(
        "dataset_preprocessors",
        "mmlu_pro_chat_preprocessor",
        desc="MMLU Pro prompt wrapper",
        tags=("prompt", "mmlu-pro"),
    )
    class MMLUPreprocessor(NewMMLUProConverter):
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


# benchmark SimpleQA Verified
@registry.asset(
    "dataset_preprocessors",
    "simpleqa_verified",
    desc="SimpleQA Verified dataset preprocessing logic",
    tags=("factuality", "simpleqa_verified", "question_answering"),
)
class SimpleQAVerifiedPreprocessorProvider(SimpleQAVerifiedPreprocessor):
    pass


# benchmark ARC-AGI-2
@registry.asset(
    "dataset_preprocessors",
    "arcagi2",
    desc="ARC-AGI-2 dataset preprocessing logic (visual abstraction and reasoning)",
    tags=("vision", "arcagi2", "reasoning", "pattern-recognition"),
)
class ARCAGI2PreprocessorProvider(ARCAGI2Preprocessor):
    pass

# benchmark ScreenSpot-Pro
@registry.asset(
    "dataset_preprocessors",
    "screenspot_pro",
    desc="ScreenSpot-Pro dataset preprocessing logic",
    tags=("vision", "screenspot-pro", "gui-grounding"),
)
class ScreenSpotProPreprocessorProvider(ScreenSpotProPreprocessor):
    pass


# benchmark CharXiv (reasoning)
@registry.asset(
    "dataset_preprocessors",
    "charxiv_reasoning",
    desc="CharXiv reasoning dataset preprocessing logic",
    tags=("vision", "charxiv", "reasoning", "chart-understanding"),
)
class CharXivReasoningPreprocessorProvider(CharXivReasoningPreprocessor):
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

# benchmark MRCR
@registry.asset(
    "dataset_preprocessors",
    "mrcr_chat_preprocessor",
    desc="MRCR prompt wrapper",
    tags=("prompt", "MRCR"),
)
class MRCRPreprocessor(MRCRConverter):
    pass


