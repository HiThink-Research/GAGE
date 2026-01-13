"""Builtin dataset resources providers exposed via the registry."""

from __future__ import annotations

from gage_eval.assets.datasets.bundles.base import BaseBundle
from gage_eval.assets.datasets.bundles.mathvista.bundle import MathVistaBundle as MathVistaBundleProvider
from gage_eval.assets.datasets.bundles.mme.bundle import MMEBundle as MMEBundleProvider
from gage_eval.registry import registry

@registry.asset(
    "bundles",
    "mathvista",
    desc="mathvista benchmark resource providers",
    tags=("caption", "ocr"),
)
class MathVistaBundle(MathVistaBundleProvider):
    pass

@registry.asset(
    "bundles",
    "mme",
    desc="MME benchmark resource providers",
    tags=("vision", "mme"),
)
class MMEBundle(MMEBundleProvider):
    pass
 