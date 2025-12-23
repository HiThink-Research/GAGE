"""Summary generator abstractions."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.evaluation.cache import EvalCache


class SummaryGenerator:
    """Base class for summary generators."""

    name = "base"

    def generate(self, cache: EvalCache) -> Optional[Dict[str, Any]]:  # pragma: no cover - abstract
        raise NotImplementedError


__all__ = ["SummaryGenerator"]
