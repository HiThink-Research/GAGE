"""Builtin dataset preprocessor implementations exposed via the registry."""

from __future__ import annotations

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.preprocessors.mme import MMEPreprocessor
from gage_eval.assets.datasets.preprocessors.mathvista import MathVistaPreprocessor
from gage_eval.assets.datasets.preprocessors.mmlu_pro import MMLUProPreprocessor

from gage_eval.registry import registry

# mathvista
@registry.asset(
    "preprocessors",
    "mathvista",
    desc="MathVista dataset preprocessing logic",
    tags=("caption", "ocr", "math"),
)
class MathVistaPreprocessorProvider(MathVistaPreprocessor):
    pass


# mme
@registry.asset(
    "preprocessors",
    "mme",
    desc="MME dataset preprocessing logic",
    tags=("vision", "mme", "multi-modal"),
)
class MMEPreprocessorProvider(MMEPreprocessor):
    pass


# mmlu_pro
@registry.asset(
    "preprocessors",
    "mmlu_pro_hf",
    desc="MMLU-Pro dataset preprocessing logic",
    tags=("nlp", "few-shot", "mmlu-pro"),
)
class MMLUProPreprocessorProvider(MMLUProPreprocessor):
    pass