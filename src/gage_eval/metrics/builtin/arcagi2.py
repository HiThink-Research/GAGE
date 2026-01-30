"""ARC-AGI-2 accuracy metric for grid prediction matching."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from loguru import logger

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.metrics.utils import extract_field, normalize_text_advanced
from gage_eval.registry import registry


def _extract_from_boxed(text: str) -> Optional[list[list[int]]]:
    """Extract JSON from LaTeX \\boxed{} command (from arc-agi-benchmarking)."""
    match = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        try:
            parsed_json = json.loads(content)
            if isinstance(parsed_json, list) and all(isinstance(row, list) for row in parsed_json):
                return parsed_json
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _backscan_json_parser(text: str) -> Optional[list[list[int]]]:
    """Extract last valid JSON by scanning backwards (from arc-agi-benchmarking)."""
    last_bracket_idx = -1
    closing_bracket = None
    for i in range(len(text) - 1, -1, -1):
        char = text[i]
        if char in (']', '}'):
            last_bracket_idx = i
            closing_bracket = char
            break

    if last_bracket_idx == -1:
        return None

    opening_bracket = '[' if closing_bracket == ']' else '{'

    bracket_counter = 1  # Start at 1 to account for the found closing bracket
    start_idx = -1

    for i in range(last_bracket_idx - 1, -1, -1):
        char = text[i]
        if char == closing_bracket:
            bracket_counter += 1
        elif char == opening_bracket:
            bracket_counter -= 1
            if bracket_counter == 0:
                start_idx = i
                break

    if start_idx == -1:
        return None

    json_candidate = text[start_idx:last_bracket_idx + 1]

    try:
        parsed_json = json.loads(json_candidate)
        # Validate the structure: must be a non-empty list of lists
        if isinstance(parsed_json, list) and parsed_json and all(isinstance(row, list) for row in parsed_json):
            return parsed_json
        else:
            return None
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_grid_from_text(text: str) -> Optional[list[list[int]]]:
    """Extract a 2D grid from text output using arc-agi-benchmarking parsing logic.

    Tries parsing strategies in the exact same order as arc-agi-benchmarking:
    1. extract_from_boxed - LaTeX \\boxed{} extraction
    2. backscan_json_parser - Reverse JSON scanning
    """
    if not text:
        return None

    text = text.strip()

    # Strategy 1: Extract from LaTeX boxed (same as arc-agi-benchmarking)
    result = _extract_from_boxed(text)
    if result is not None:
        return result

    # Strategy 2: Backscan JSON parser (same as arc-agi-benchmarking)
    result = _backscan_json_parser(text)
    if result is not None:
        return result

    return None


def _grids_equal(pred: list[list[int]], ref: list[list[int]]) -> bool:
    """Check if two grids are equal."""
    if not isinstance(pred, list) or not isinstance(ref, list):
        return False
    if len(pred) != len(ref):
        return False
    for pred_row, ref_row in zip(pred, ref):
        if not isinstance(pred_row, list) or not isinstance(ref_row, list):
            return False
        if len(pred_row) != len(ref_row):
            return False
        if pred_row != ref_row:
            return False
    return True


@registry.asset(
    "metrics",
    "arcagi2_accuracy",
    desc="ARC-AGI-2 grid prediction accuracy",
    tags=("arcagi2", "grid", "accuracy"),
    default_aggregation="mean",
)
class ARCAGI2AccuracyMetric(SimpleMetric):
    """Metric for evaluating ARC-AGI-2 grid predictions."""

    value_key = "accuracy"

    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: Resolve config fields.
        prediction_field = self.args.get("prediction_field", "model_output.answer")
        reference_field = self.args.get("reference_field", "sample.references")

        # STEP 2: Extract prediction and reference
        prediction_raw = extract_field(context, prediction_field, default="")
        reference_raw = extract_field(context, reference_field, default=None)

        # Normalize prediction to string
        if prediction_raw is None:
            prediction_str = ""
        else:
            prediction_str = str(prediction_raw)

        # Extract reference grid
        reference_grid = None
        if reference_raw is not None:
            if isinstance(reference_raw, list):
                # Handle list of references - use the first one
                if len(reference_raw) > 0:
                    first_ref = reference_raw[0]
                    if isinstance(first_ref, list) and all(isinstance(row, list) for row in first_ref):
                        # First element is already a grid
                        reference_grid = first_ref
                    elif isinstance(first_ref, list):
                        reference_grid = first_ref
                else:
                    reference_grid = reference_raw
            elif isinstance(reference_raw, str):
                try:
                    reference_grid = json.loads(reference_raw)
                except (json.JSONDecodeError, ValueError):
                    reference_grid = _extract_grid_from_text(reference_raw)
            elif isinstance(reference_raw, dict):
                # Handle dict with 'test' key containing test outputs
                if "test" in reference_raw and isinstance(reference_raw["test"], list):
                    # Get the first test output
                    test_data = reference_raw["test"][0]
                    if isinstance(test_data, dict) and "output" in test_data:
                        reference_grid = test_data["output"]
                    else:
                        reference_grid = test_data

        # Extract prediction grid
        prediction_grid = _extract_grid_from_text(prediction_str)

        # STEP 3: Compare grids
        is_correct = False
        metadata: Dict[str, Any] = {
            "prediction_raw": prediction_str[:500] if prediction_str else "",
            "reference_grid": reference_grid,
            "prediction_grid": prediction_grid,
        }

        if reference_grid is None:
            metadata["error"] = "No reference grid found"
            return MetricResult(
                sample_id=context.sample_id,
                values={self.value_key: 0.0},
                metadata=metadata,
            )

        if prediction_grid is None:
            metadata["error"] = "Failed to extract prediction grid from output"
            return MetricResult(
                sample_id=context.sample_id,
                values={self.value_key: 0.0},
                metadata=metadata,
            )

        is_correct = _grids_equal(prediction_grid, reference_grid)
        metadata["is_correct"] = is_correct

        return MetricResult(
            sample_id=context.sample_id,
            values={self.value_key: 1.0 if is_correct else 0.0},
            metadata=metadata,
        )


__all__ = ["ARCAGI2AccuracyMetric"]