"""AMO-Bench accuracy metric.

AMO-Bench uses different evaluation methods based on answer_type:
- "description": Uses LLM judge (from sample["eval_result"]["answer"])
- "number"/"set": Uses math_verify parser
- "variable": Uses try_list with sympy solver
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.metrics.utils import (
    extract_field,
    get_text_content_of_first_predict_result,
    get_first_reference,
)
from gage_eval.registry import registry

# Try to import optional dependencies
try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    parse = None
    verify = None

try:
    from sympy import solve
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    solve = None


# Answer extraction patterns
ANSWER_PREFIX_LIST = [
    "### the final answer is:", "### the final answer:", "### final answer is:", "### final answer:",
    "### the final answer is", "### the final answer", "### final answer is", "### final answer",
]
ANSWER_PREFIX_LIST_WO_HASHTAG = [p[4:] for p in ANSWER_PREFIX_LIST]

THINK_POSTFIX_LIST = [
    "</think>",
    "</longcat_think>",
]

REMOVE_LIST = [
    "\\bigl", "\\bigr", 
    "\\Bigl", "\\Bigr",
    "\\biggl", "\\biggr",
    "\\Biggl", "\\Biggr",
    "\\bigg", "\\Bigg", "\\big", "\\Big",
    "\\left", "\\right",
]

REPLACE_LIST = [
    ("'", "'"),
    ("'", "'"),
    ('"', '"'),
    ('"', '"'),
    ("(", "("),
    (")", ")"),
    (", ", ", "),
    (": ", ": "),
    ("; ", "; "),
    ("。", ". "),
    ("！", "! "),
    ("？", "? "),
    ("…", "..."),
    ("–", "-"),
    ("−", "-"),
]


def _pred_extractor(pred: str, answer_type: str) -> str:
    """Extract answer from prediction text.
    
    Ported from AMO-Bench utils.py.
    """
    pred_extract = pred.replace('：', ': ')
    
    for think_postfix in THINK_POSTFIX_LIST:
        pred_extract = pred_extract.split(think_postfix)[-1].strip()
    
    for prefix in ANSWER_PREFIX_LIST + ANSWER_PREFIX_LIST_WO_HASHTAG:
        if prefix in pred_extract.lower():
            pred_extract_lower = pred_extract.lower().split(prefix)[-1]
            pred_extract = pred_extract[-len(pred_extract_lower):]
            pred_extract = pred_extract.strip()
            break
    
    if answer_type != "description":
        for pat in REMOVE_LIST:
            pred_extract = pred_extract.replace(pat, "")
    
    for pat, new_pat in REPLACE_LIST:
        pred_extract = pred_extract.replace(pat, new_pat)
    
    while " }" in pred_extract:
        pred_extract = pred_extract.replace(" }", "}")
    while ".}" in pred_extract:
        pred_extract = pred_extract.replace(".}", "}")
    
    if answer_type in ["number", "variable", "set"]:
        pred_extract = pred_extract.replace(r"\,", "")
        pred_extract = pred_extract.replace(r"\;", "")
        pred_extract = pred_extract.replace("\ ", " ")
        pred_extract = pred_extract.replace("\;", ";")
        pred_extract = pred_extract.replace("\n", " ")
    
    if answer_type in ["number", "variable"]:
        pred_extract = pred_extract.replace(",", "")
        pred_extract = pred_extract.replace("\\{", "(").replace("\\}", ")").replace("\\[", "(").replace("\\]", ")")
    
    return pred_extract.strip()


def _extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from \\boxed{...} pattern."""
    if not text:
        return None
    
    # Find \\boxed{ pattern
    pattern = r"\\boxed\s*\{"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    
    if not matches:
        return None
    
    # Process each match and extract content with balanced braces
    extracted_contents = []
    for match in matches:
        start_idx = match.end()
        brace_count = 1
        end_idx = start_idx
        
        while end_idx < len(text) and brace_count > 0:
            if text[end_idx] == "{":
                brace_count += 1
            elif text[end_idx] == "}":
                brace_count -= 1
            end_idx += 1
        
        if brace_count == 0:
            # Successfully found matching braces
            content = text[start_idx:end_idx - 1].strip()
            extracted_contents.append(content)
    
    if extracted_contents:
        return extracted_contents[-1]
    return None


def _verify_number_set_answer(pred_extract: str, gold_answer: str) -> bool:
    """Verify number or set type answers using math_verify.
    
    Args:
        pred_extract: Extracted prediction string.
        gold_answer: Gold answer string.
        
    Returns:
        True if answers match, False otherwise.
    """
    if not MATH_VERIFY_AVAILABLE:
        # Fallback to string comparison if math_verify not available
        return pred_extract.strip() == gold_answer.strip()
    
    try:
        pred_parse = parse(pred_extract)
        gold_parse = parse(gold_answer)
        verify_result = verify(gold_parse, pred_parse, float_rounding=4) or verify(pred_parse, gold_parse, float_rounding=4)
        
        # Try extracting from last equation if present
        if pred_parse and '=' in str(pred_parse[-1]):
            pred_last_str = str(pred_parse[-1]).split('=')[-1]
            pred_last_str_parse = parse("\\boxed{" + pred_last_str + "}")
            verify_last_result = verify(gold_parse, pred_last_str_parse, float_rounding=4) or verify(pred_last_str_parse, gold_parse, float_rounding=4)
            verify_result = verify_result or verify_last_result
        
        return verify_result
    except Exception:
        return False


def _verify_variable_answer(pred_extract: str, gold_answer: str, try_list: List[str]) -> bool:
    """Verify variable type answers using try_list.
    
    Args:
        pred_extract: Extracted prediction string.
        gold_answer: Gold answer string.
        try_list: List of test values (e.g., ["n=1", "n=2", ...]).
        
    Returns:
        True if answers match, False otherwise.
    """
    if not MATH_VERIFY_AVAILABLE or not SYMPY_AVAILABLE or not try_list:
        return False
    
    try:
        pred_parse_ori = parse(pred_extract)
        if not pred_parse_ori:
            return False
        
        pred_parse_str = str(pred_parse_ori[-1])
        pred_parse_str = pred_parse_str.split("\\qquad")[-2].strip() if "\\qquad" in pred_parse_str else pred_parse_str
        pred_parse_str = pred_parse_str.split("\\quad")[-2].strip() if "\\quad" in pred_parse_str else pred_parse_str
        pred_parse_str = pred_parse_str.split("=")[-1]
        
        gold_parse_ori = parse(gold_answer)
        if not gold_parse_ori:
            return False
        gold_parse_str = str(gold_parse_ori[-1])
        gold_parse_str = gold_parse_str.split("=")[-1]
        
        for try_str in try_list:
            pred_parse_equ = parse("\\boxed{" + try_str + ", y=" + pred_parse_str + "}")
            gold_parse_equ = parse("\\boxed{" + try_str + ", y=" + gold_parse_str + "}")
            
            try:
                pred_parse_solve = solve(pred_parse_equ[0])
            except Exception:
                return False
            
            gold_parse_solve = solve(gold_parse_equ[0])
            
            if not gold_parse_solve:
                return False
            
            if not pred_parse_solve:
                return False
            
            if isinstance(pred_parse_solve, list):
                pred_parse_solve = pred_parse_solve[0]
            if isinstance(gold_parse_solve, list):
                gold_parse_solve = gold_parse_solve[0]
            
            pred_parse_solve_y = None
            gold_parse_solve_y = None
            
            try:
                for s in pred_parse_solve:
                    if str(s) == 'y':
                        pred_parse_solve_y = pred_parse_solve[s]
            except Exception:
                return False
            
            for s in gold_parse_solve:
                if str(s) == 'y':
                    gold_parse_solve_y = gold_parse_solve[s]
            
            if gold_parse_solve_y is None:
                return False
            
            pred_parse_solve_y = pred_parse_solve_y.evalf()
            gold_parse_solve_y = gold_parse_solve_y.evalf()
            
            from math_verify import verify as mv_verify
            verify_result = mv_verify(gold_parse_solve_y, pred_parse_solve_y, float_rounding=8) or mv_verify(pred_parse_solve_y, gold_parse_solve_y, float_rounding=8)
            
            if not verify_result:
                return False
        
        return True
    except Exception:
        return False


def _extract_judge_verdict(eval_result_answer: str) -> bool:
    """Extract correctness verdict from judge model output.
    
    This function extracts the conclusion from the judge model's response
    and determines if the answer is correct based on the conclusion.
    
    The logic follows AMO-Bench's utils.py:
    - Extract text after "### Conclusion:"
    - Check if "correct" appears as a standalone word
    - Exclude cases where it's "not correct" or "n't correct"
    
    Args:
        eval_result_answer: The judge model's answer text from eval_result.
        
    Returns:
        True if judge determines answer is correct, False otherwise.
    """
    if not eval_result_answer or not isinstance(eval_result_answer, str):
        return False
    
    # Extract conclusion part after "### Conclusion:"
    conclusion = eval_result_answer.lower().split("conclusion:")[-1]
    
    # Check if "correct" is in the conclusion as a standalone word,
    # but exclude "not correct" and "n't correct" cases
    if "correct" in conclusion.split() and "not correct" not in conclusion and "n't correct" not in conclusion:
        return True
    return False


@registry.asset(
    "metrics",
    "amo_bench_accuracy",
    desc="AMO-Bench accuracy metric with different evaluation methods based on answer_type",
    tags=("amo_bench", "math"),
    default_aggregation="mean",
)
class AMOBenchAccuracyMetric(SimpleMetric):
    """Metric for evaluating AMO-Bench answers.
    
    This metric automatically handles different answer types:
    - "description": Uses judge model output (from sample["eval_result"]["answer"])
    - "number"/"set": Uses math_verify parser for mathematical equivalence
    - "variable": Uses try_list with sympy solver for function verification
    """
    
    value_key = "acc"
    
    def compute(self, context: MetricContext) -> MetricResult:
        """Compute AMO-Bench accuracy metric.
        
        Args:
            context: The metric context containing sample, prediction, and judge data.
            
        Returns:
            MetricResult with accuracy score and metadata.
        """
        # STEP 1: Extract sample info
        sample_dict = extract_field(context, "sample")
        answer = get_first_reference(sample_dict)
        prediction_raw = get_text_content_of_first_predict_result(sample_dict)
        
        metadata = dict(sample_dict.get("metadata") or {})
        answer_type = metadata.get("answer_type", "")
        try_list = metadata.get("try_list", [])
        
        # STEP 2: Compute score based on answer type
        is_correct = False
        pred_extract = ""
        
        if answer_type == "description":
            # For description type: use judge output from sample's eval_result
            # The judge model output is stored in sample["eval_result"]["answer"]
            eval_result = dict(sample_dict.get("eval_result") or {})
            judge_answer = eval_result.get("answer", "")
            is_correct = _extract_judge_verdict(judge_answer)
            # Extract prediction for metadata
            pred_extract = _pred_extractor(prediction_raw, answer_type) if prediction_raw else ""
            boxed_content = _extract_boxed_content(prediction_raw) if prediction_raw else None
            if boxed_content:
                pred_extract = boxed_content
                
        elif answer_type in ["number", "set"]:
            # For number/set types: use math_verify
            pred_extract = _pred_extractor(prediction_raw, answer_type) if prediction_raw else ""
            is_correct = _verify_number_set_answer(pred_extract, str(answer))
            # Try with boxed content as fallback
            if not is_correct:
                boxed_content = _extract_boxed_content(prediction_raw) if prediction_raw else None
                if boxed_content:
                    is_correct = _verify_number_set_answer(boxed_content, str(answer))
                    if is_correct:
                        pred_extract = boxed_content
                        
        elif answer_type == "variable":
            # For variable type: use try_list with sympy
            pred_extract = _pred_extractor(prediction_raw, answer_type) if prediction_raw else ""
            is_correct = _verify_variable_answer(pred_extract, str(answer), try_list)
            # Try with boxed content as fallback
            if not is_correct:
                boxed_content = _extract_boxed_content(prediction_raw) if prediction_raw else None
                if boxed_content:
                    is_correct = _verify_variable_answer(boxed_content, str(answer), try_list)
                    if is_correct:
                        pred_extract = boxed_content
        else:
            # Unknown type: fallback to exact string match
            pred_extract = _pred_extractor(prediction_raw, answer_type) if prediction_raw else ""
            is_correct = pred_extract.strip() == str(answer).strip()
        
        score = 1.0 if is_correct else 0.0
        
        result_metadata = {
            "prediction": pred_extract,
            "references": answer,
            "answer_type": answer_type,
            "eval_method": "judge" if answer_type == "description" else "parser",
        }
        
        if try_list:
            result_metadata["try_list"] = try_list
        
        if answer_type == "description":
            eval_result = dict(sample_dict.get("eval_result") or {})
            if eval_result:
                result_metadata["judge_verdict"] = is_correct
                result_metadata["judge_answer"] = eval_result.get("answer", "")
        
        return MetricResult(
            sample_id=context.sample_id,
            values={self.value_key: score},
            metadata=result_metadata,
        )


__all__ = [
    "AMOBenchAccuracyMetric",
    "_pred_extractor",
    "_extract_boxed_content",
    "_verify_number_set_answer",
    "_verify_variable_answer",
    "_extract_judge_verdict",
]
