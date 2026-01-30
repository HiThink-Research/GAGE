"""Builtin dataset resources providers exposed via the registry."""

from __future__ import annotations

from gage_eval.assets.datasets.bundles.base import BaseBundle
from gage_eval.assets.datasets.bundles.mathvista.bundle import MathVistaBundle
from gage_eval.assets.datasets.bundles.mmlu_pro.few_shot import MMLUProBundle
from gage_eval.assets.datasets.bundles.mme import MMEBundle
from gage_eval.assets.datasets.bundles.screenspot_pro.bundle import ScreenSpotProBundle
from gage_eval.assets.datasets.bundles.charxiv.bundle import CharXivBundle
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

# mme
@registry.asset(
    "bundles",
    "mme",
    desc="MME benchmark resource providers",
    tags=("vision", "mme"),
)
class MMEBundleProvider(MMEBundle):
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

# screenspot_pro
@registry.asset(
    "bundles",
    "screenspot_pro",
    desc="ScreenSpot-Pro benchmark resource providers",
    tags=("vision", "screenspot-pro", "gui-grounding"),
)
class ScreenSpotProBundleProvider(ScreenSpotProBundle):
    pass

# charxiv
@registry.asset(
    "bundles",
    "charxiv",
    desc="CharXiv benchmark resource providers",
    tags=("vision", "charxiv", "chart-understanding"),
)
class CharXivBundleProvider(CharXivBundle):
    pass