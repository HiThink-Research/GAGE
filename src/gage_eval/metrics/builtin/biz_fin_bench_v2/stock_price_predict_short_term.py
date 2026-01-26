"""FizBinBench stock_price_predict accuracy metric"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Literal

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric

from gage_eval.metrics.filter.base import RegexFilter
from gage_eval.metrics.numeric import extract_numeric_answer
from gage_eval.metrics.match import match_str
from gage_eval.metrics.choice import extract_single_choice_letter

from gage_eval.metrics.utils import (
    ensure_list_of_strings, extract_field, normalize_text_advanced,
        get_text_content_of_first_predict_result,
    get_sample_label,
    get_first_reference
)

from gage_eval.registry import registry
import json
import re
from typing import Optional, Tuple, Dict, Any

# Compatible with { } and {{ }} braces
_OPEN_BRACE = r'\{\{?'
_CLOSE_BRACE = r'\}\}?'
_NUM = r'[-+]?\d+(?:\.\d+)?'

def _tail(text: str, n: int = 8000) -> str:
    """Take last n chars to reduce false matches from prompt/examples."""
    if not isinstance(text, str):
        return ""
    return text[-n:]


def _extract_answer_boxed_value(predict_result: str) -> Optional[float]:
    """
    Extract predicted value from answer_boxed{...} or answer_boxed{{...}}.
    Parse from back to front: take the LAST occurrence.
    """
    if not isinstance(predict_result, str):
        return None

    text = predict_result.strip()
    tail = _tail(text, 8000)

    pattern = re.compile(
        rf'answer_boxed\s*{_OPEN_BRACE}\s*({_NUM})\s*{_CLOSE_BRACE}',
        re.IGNORECASE
    )
    matches = list(pattern.finditer(tail))
    if matches:
        try:
            return float(matches[-1].group(1))
        except Exception:
            return None

    # Fallback: last number near the end
    tail2 = _tail(text, 300)
    nums = re.findall(r'[-+]?\d+(?:\.\d+)?', tail2)
    for s in reversed(nums):
        try:
            return float(s)
        except ValueError:
            continue
    return None


def _extract_interval_boxed_values(predict_result: str) -> Optional[Tuple[float, float]]:
    """
    Extract interval from interval_boxed{[l,u]} or interval_boxed{{[l,u]}}.
    Parse from back to front: take the LAST occurrence.
    """
    if not isinstance(predict_result, str):
        return None

    text = predict_result.strip()
    tail = _tail(text, 8000)

    pattern = re.compile(
        rf'interval_boxed\s*{_OPEN_BRACE}\s*\[\s*({_NUM})\s*,\s*({_NUM})\s*\]\s*{_CLOSE_BRACE}',
        re.IGNORECASE | re.DOTALL
    )
    matches = list(pattern.finditer(tail))
    if matches:
        m = matches[-1]
        try:
            lower = float(m.group(1))
            upper = float(m.group(2))
            if lower > upper:
                lower, upper = upper, lower
            return (lower, upper)
        except Exception:
            return None

    # Fallback: last [a,b] near end
    tail2 = _tail(text, 600)
    bracket_pattern = re.compile(rf'\[\s*({_NUM})\s*,\s*({_NUM})\s*\]')
    bracket_matches = list(bracket_pattern.finditer(tail2))
    if bracket_matches:
        m = bracket_matches[-1]
        try:
            lower = float(m.group(1))
            upper = float(m.group(2))
            if lower > upper:
                lower, upper = upper, lower
            return (lower, upper)
        except Exception:
            return None

    return None

def _extract_float_from_text(text: str) -> Optional[float]:
    """Extract the first float/integer from text."""
    if not isinstance(text, str):
        return None

    clean_text = re.sub(r'(?<=\d),(?=\d)', '', text)
    clean_text = re.sub(r'[$,¥€£%]', '', clean_text)

    for pattern in [r'[-+]?\d+\.\d+', r'[-+]?\d+']:
        m = re.search(pattern, clean_text)
        if m:
            try:
                return float(m.group())
            except ValueError:
                pass
    return None

def _validate_output_format(predict_result: str) -> Dict[str, Any]:
    """
    Validate last two non-empty lines contain answer_boxed + interval_boxed
    (supports {} and {{}}).
    """
    if not isinstance(predict_result, str):
        return {
            "is_valid": False,
            "errors": ["predict_result is not a string"],
            "has_answer_boxed": False,
            "has_interval_boxed": False,
            "line_count": 0,
            "predicted_value": None,
            "interval": None,
            "tail_lines": []
        }

    text = predict_result.strip()
    tail = _tail(text, 8000)

    all_lines = [line.strip() for line in tail.splitlines() if line.strip()]
    tail_lines = all_lines[-2:] if len(all_lines) >= 2 else all_lines
    line_count = len(tail_lines)
    tail_joined = "\n".join(tail_lines)

    has_answer_boxed = bool(
        re.search(rf'answer_boxed\s*{_OPEN_BRACE}\s*{_NUM}\s*{_CLOSE_BRACE}', tail_joined, re.IGNORECASE)
    )
    has_interval_boxed = bool(
        re.search(
            rf'interval_boxed\s*{_OPEN_BRACE}\s*\[\s*{_NUM}\s*,\s*{_NUM}\s*\]\s*{_CLOSE_BRACE}',
            tail_joined,
            re.IGNORECASE | re.DOTALL
        )
    )

    predicted_value = _extract_answer_boxed_value(text)
    interval = _extract_interval_boxed_values(text)

    errors = []
    if line_count != 2:
        errors.append(f"Output should have exactly 2 lines at the end, but tail has {line_count}")
    if not has_answer_boxed:
        errors.append("Missing or malformed answer_boxed{} (or answer_boxed{{}}) in the last two lines")
    if not has_interval_boxed:
        errors.append("Missing or malformed interval_boxed{} (or interval_boxed{{}}) in the last two lines")
    if predicted_value is None:
        errors.append("Could not extract predicted value from the LAST answer_boxed")
    if interval is None:
        errors.append("Could not extract interval from the LAST interval_boxed")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "has_answer_boxed": has_answer_boxed,
        "has_interval_boxed": has_interval_boxed,
        "line_count": line_count,
        "predicted_value": predicted_value,
        "interval": interval,
        "tail_lines": tail_lines
    }

def _check_interval_is_0p99_1p01_of_answer(
    answer_value: float,
    interval: Tuple[float, float],
    *,
    lo_mul: float = 0.99,
    hi_mul: float = 1.01,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
) -> Dict[str, Any]:
    """
    Strict rule (1% CI):
    interval must be exactly [answer_value*0.99, answer_value*1.01] (allow tiny float tol).
    """
    lower, upper = interval
    if lower > upper:
        lower, upper = upper, lower

    e1 = answer_value * lo_mul
    e2 = answer_value * hi_mul
    exp_low, exp_high = (e1, e2) if e1 <= e2 else (e2, e1)

    def _close(a: float, b: float) -> bool:
        return abs(a - b) <= max(abs_tol, rel_tol * max(1.0, abs(b)))

    errors = []
    if not _close(lower, exp_low):
        errors.append(f"interval lower != answer*{lo_mul} (got {lower}, expected {exp_low})")
    if not _close(upper, exp_high):
        errors.append(f"interval upper != answer*{hi_mul} (got {upper}, expected {exp_high})")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "expected_interval": (exp_low, exp_high),
        "given_interval": (lower, upper),
    }



@registry.asset(
    "metrics",
    "stock_price_predict_short_term", 
    desc="stock_price_predict_short_term",
    tags=("BizFinBench V2",),
    default_aggregation="mean",
)
class StockPricePredictAccuracyMetric(SimpleMetric):
    value_key = "acc"
    regex_pattern = r"The best answer is:\s*(.+?)"
    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: extract sample/predict /groud truth
        sample_dict = extract_field(context, 'sample')
        answer = get_first_reference(sample_dict)
        true_value = _extract_float_from_text(answer)
        predict_result = get_text_content_of_first_predict_result(sample_dict)

        format_validation = _validate_output_format(predict_result)
        answer_value = format_validation["predicted_value"]  # = answer_boxed{}
        interval = format_validation["interval"]

        score = 0.0
        result = "False"

        parse_successful = (
            true_value is not None and
            interval is not None and
            answer_value is not None
        )

        interval_check = {"ok": False, "errors": ["not_checked"], "expected_interval": None, "given_interval": None}
        score = 0.0
        if parse_successful:
            # 1) enforce interval == [0.99*answer, 1.01*answer]
            interval_check = _check_interval_is_0p99_1p01_of_answer(
                    answer_value, interval,
                    lo_mul=0.99, hi_mul=1.01,
                    rel_tol=1e-9, abs_tol=1e-9
            )

            if interval_check["ok"]:
                lower, upper = interval
                if lower > upper:
                    lower, upper = upper, lower

                # 2) only if true_value is within the interval => score=1
                if lower <= true_value <= upper:
                    score = 1.0

        # STEP 3: compute score
        metadata = {"prediction": predict_result, "references": answer}
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)

__all__ = ["StockPricePredictAccuracyMetric", ]

if __name__ ==  '__main__':
    from gage_eval.config.pipeline_config import MetricSpec
    spec = MetricSpec(metric_id='test', implementation='fake_acc')
    global_metric = StockPricePredictAccuracyMetric(spec=spec)
