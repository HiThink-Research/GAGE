"""CharXiv-specific metrics."""

from __future__ import annotations

import re
from typing import Any

from gage_eval.metrics.base import ComparisonMetric
from gage_eval.metrics.utils import normalize_text_advanced
from gage_eval.registry import registry


@registry.asset(
    "metrics",
    "charxiv_reasoning_match",
    desc="CharXiv reasoning answer match allowing option-only or content-only outputs",
    tags=("text", "charxiv"),
    default_aggregation="mean",
)
class CharXivReasoningMatchMetric(ComparisonMetric):
    """Match metric for CharXiv reasoning.

    Treats the following forms as equivalent correct answers:
      - full annotated answer: "(b) OPT"
      - option only: "(b)" or "b"
      - content only: "OPT"
    """

    default_reference_field = "sample.label"
    default_prediction_field = "model_output.answer"

    def compare(self, prediction: Any, reference: Any) -> tuple[float, dict]:
        case_sensitive = bool(self.args.get("case_sensitive", False))
        strip = bool(self.args.get("strip_whitespace", True))

        ref_norm = normalize_text_advanced(
            reference,
            case_sensitive=case_sensitive,
            strip=strip,
            collapse_whitespace=True,
        ) or ""
        pred_norm = normalize_text_advanced(
            prediction,
            case_sensitive=case_sensitive,
            strip=strip,
            collapse_whitespace=True,
        ) or ""

        if not ref_norm or not pred_norm:
            return 0.0, {}

        # 1) direct exact match
        if ref_norm == pred_norm:
            return 1.0, {}

        # 2) parse annotated reference like "(b) OPT"
        m = re.match(r"^\(?([A-Za-z])\)?\s*(.+)$", ref_norm)
        if not m:
            # Fallback: strict match only
            return 0.0, {}

        opt = m.group(1).strip()
        core = m.group(2).strip()
        opt_paren = f"({opt})"

        # Accept multiple equivalent forms
        acceptable = {
            ref_norm,   # "(b) OPT"
            core,       # "OPT"
            opt,        # "b"
            opt_paren,  # "(b)"
        }

        score = 1.0 if pred_norm in acceptable else 0.0
        return score, {
            "normalized_reference": ref_norm,
            "normalized_prediction": pred_norm,
            "core": core,
            "option": opt,
        }

