"""FizBinBench eval_financial_description_no_cot accuracy metric"""

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

# { } and {{ }} 
_OPEN_BRACE = r'\{\{?'
_CLOSE_BRACE = r'\}\}?'

_NUM = r'[-+]?\d+(?:\.\d+)?'  # number


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


def _get_true_value_from_sample(sample: dict) -> Optional[float]:
    """Extract true value from common fields; prefers choices[0].message.content."""
    choices = sample.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content", [])

        if isinstance(content, list) and content and isinstance(content[0], dict):
            choice_text = content[0].get("text", "")
        elif isinstance(content, str):
            choice_text = content
        else:
            choice_text = ""

        v = _extract_float_from_text(choice_text)
        if v is not None:
            return v

    true_value = sample.get("true_value")
    if true_value is not None:
        if isinstance(true_value, (int, float)):
            return float(true_value)
        if isinstance(true_value, str):
            return _extract_float_from_text(true_value)

    choice_text = sample.get("choice_text", "")
    if choice_text:
        v = _extract_float_from_text(choice_text)
        if v is not None:
            return v

    for field in ["ground_truth", "target", "actual", "actual_value"]:
        if field in sample:
            fv = sample[field]
            if isinstance(fv, (int, float)):
                return float(fv)
            if isinstance(fv, str):
                v = _extract_float_from_text(fv)
                if v is not None:
                    return v

    return None


def _validate_output_format(predict_result: str) -> Dict[str, Any]:
    """
    Validate last two non-empty lines contain answer_boxed + interval_boxed (supports {} and {{}}).
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


def _check_interval_is_0p90_1p10_of_answer(
    answer_value: float,
    interval: Tuple[float, float],
    *,
    lo_mul: float = 0.90,
    hi_mul: float = 1.10,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
) -> Dict[str, Any]:
    """
    Strict rule:
    interval must be exactly [answer_value*0.90, answer_value*1.10] (allow tiny float tol).
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
    "eval_emotion",
    desc="eval_emtion",
    tags=("BizFinBench V2",),
    default_aggregation="mean",
)
class EvalEmotionAccuracyMetric(SimpleMetric):
    value_key = "acc"
    def compute(self, context: MetricContext) -> MetricResult:
        score = 0
        metadata = {}
        correct_predictions = 0
        successful_parses = 0
        failed_parses = 0
        correct_on_successful = 0
        try:
            # STEP 1: extract sample/predict /groud truth
            sample_dict = extract_field(context, 'sample')
            ans_text = get_first_reference(sample_dict)
            true_value = _extract_float_from_text(ans_text)
            predict_result_str = get_text_content_of_first_predict_result(sample_dict)
            format_validation = _validate_output_format(predict_result_str)
            answer_value = format_validation["predicted_value"]  # 就是 answer_boxed 里的值
            interval = format_validation["interval"]

            score = 0.0
            result = "False"

            parse_successful = (
                true_value is not None and
                interval is not None and
                answer_value is not None
            )

            interval_check = {"ok": False, "errors": ["not_checked"], "expected_interval": None, "given_interval": None}

            if parse_successful:
                successful_parses += 1

                # 1) 必须满足 interval == [answer*0.90, answer*1.10]
                interval_check = _check_interval_is_0p90_1p10_of_answer(
                    answer_value, interval,
                    lo_mul=0.90, hi_mul=1.10,
                    rel_tol=1e-9, abs_tol=1e-9
                )

                if interval_check["ok"]:
                    lower, upper = interval
                    if lower > upper:
                        lower, upper = upper, lower

                    # 2) true_value 落在区间内 => score=1
                    if lower <= true_value <= upper:
                        score = 1.0
                        result = "True"
                        correct_predictions += 1
                        correct_on_successful += 1
            else:
                failed_parses += 1

            metadata["eval_result"] = {
                "result": result,
                "true_value": true_value,
                "predicted_value": answer_value,
                "interval": interval,
                "parse_successful": parse_successful,
                "format_is_valid": format_validation.get("is_valid", False),
                "format_errors": format_validation.get("errors", []),
                "parsed_tail_lines": format_validation.get("tail_lines", []),
                "interval_matches_0.90_1.10": interval_check.get("ok", False),
                "interval_multiplier_errors": interval_check.get("errors", []),
                "expected_interval": interval_check.get("expected_interval"),
                "given_interval_normalized": interval_check.get("given_interval"),
            }
        except Exception as e:
            failed_parses += 1
            metadata["eval_result"] = {
                "result": "False",
                "error": str(e),
                "true_value": None,
                "predicted_value": None,
                "interval": None,
                "parse_successful": False
            }

        # STEP 3: compute score
        #metadata.update({"prediction": predicted_answers, "references": correct_answers})
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)

__all__ = ["EvalEmotionAccuracyMetric", ]

if __name__ ==  '__main__':
    from gage_eval.config.pipeline_config import MetricSpec
    spec = MetricSpec(metric_id='test', implementation='fake_acc')
    global_metric = EvalEmotionAccuracyMetric(spec=spec)
    context = MetricContext(
            sample_id="demo",
            sample={"predict_result": [{"message": {"content": [{"text": 'answer_boxed{50.0}\ninterval_boxed{[45,55]}'
}]}}],
                    "references": ["50"]
            },
            model_output={
            },
            judge_output={},
            args={},
            trace=None,
        )
    ret = global_metric.compute(context)
    print("ret:", ret)
    