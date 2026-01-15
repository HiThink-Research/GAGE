"""Builtin dataset resources providers exposed via the registry."""

from __future__ import annotations

from gage_eval.assets.datasets.bundles.base import BaseBundle
from gage_eval.assets.datasets.bundles.mathvista.bundle import MathVistaBundle 
from gage_eval.assets.datasets.bundles.mmlu_pro.few_shot import MMLUProBundle
from gage_eval.registry import registry

# mathvista
@registry.asset(
    "bundles",
    "mathvista",
    desc="mathvista benchmark resource providers",
    tags=("caption", "ocr"),
)
class MathVistaBundleProvider(MathVistaBundle):
    pass

# mmlu_pro
@registry.asset(
    "bundles",
    "mmlu_pro_hf",
    desc="mmlu-pro benchmark resource providers",
    tags=("mmlu-pro", "few-shot"),
)
class MMLUProBundleProvider(MMLUProBundle):
    pass