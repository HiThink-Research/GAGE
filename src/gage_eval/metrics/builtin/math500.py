"""MATH-500 accuracy metric for LaTeX answer matching using PRM800K grading logic."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from loguru import logger

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.metrics.utils import (
    ensure_list_of_strings,
    extract_field,
    normalize_text_advanced,
    get_text_content_of_first_predict_result,
    get_first_reference,
)
from gage_eval.registry import registry

# Try to import optional dependencies for PRM800K grading
try:
    import sympy
    from sympy.parsing import sympy_parser
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sympy = None
    sympy_parser = None

try:
    from pylatexenc import latex2text
    LATEX2TEXT_AVAILABLE = True
except ImportError:
    LATEX2TEXT_AVAILABLE = False
    latex2text = None

_BOXED_PATTERN = re.compile(r"\\boxed\s*\{([^}]*)\}", re.IGNORECASE)
# Pattern to match LaTeX math expressions: \(...\) or \[...\] or \left(...\right)
_LATEX_MATH_PATTERN = re.compile(
    r"\\(?:\(|\[|left\s*\(|begin\{equation\})\s*([^\)\]}]+)\s*\\(?:\)|\]|right\)|end\{equation\})",
    re.DOTALL | re.IGNORECASE
)
# Pattern to match "the answer is" or similar phrases
_ANSWER_PHRASE_PATTERN = re.compile(
    r"(?:the\s+)?(?:answer|result|solution|final\s+answer|coordinates?|are|is)\s+[:\-]?\s*([^\n\.]+)",
    re.IGNORECASE
)
# Pattern to match content after equals sign in math context: = (3, 0) or = \frac{1}{2}
_EQUALS_PATTERN = re.compile(r"=\s*([^\n]+?)(?:\\\]|\\\)|$)", re.IGNORECASE)
# Pattern to match coordinate tuples: (r, θ) = (3, 0) or (3, \frac{\pi}{2})
_COORDINATE_PATTERN = re.compile(r"\([^\)]*\)\s*=\s*(\([^\)]+\))", re.IGNORECASE)

# PRM800K grading constants
_BAD_SUBSTRINGS = ["^{", "^("]
_BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
_TUPLE_CHARS = "()[]"


def _extract_boxed_answer(text: str) -> Optional[str]:
    """Extract LaTeX answer from \\boxed{...} pattern."""
    if not text:
        return None
    matches = _BOXED_PATTERN.findall(text)
    if matches:
        # Return the last match (most likely the final answer)
        return matches[-1].strip()
    return None


def _extract_answer_from_text(text: str) -> Optional[str]:
    """Extract answer from text when \\boxed{} is not present.
    
    Tries multiple strategies:
    1. Extract from "the answer is ..." phrases
    2. Extract \left(...\right) expressions (handles nested parentheses)
    3. Extract last LaTeX math expression \(...\) or \[...\] or \left(...\right)
    4. Extract content after equals sign in coordinate context: (r, θ) = (3, 0)
    5. Extract content after equals sign in math context: = \frac{1}{2}
    6. If text is a pure LaTeX expression, return as-is
    7. Extract last line if it contains math-like content
    """
    if not text:
        return None
    
    def _clean_candidate(c: str) -> Optional[str]:
        """Clean and validate candidate answer."""
        c = re.sub(r'[\.\,\;\:]+$', '', c.strip())
        return c if c and len(c) < 500 else None
    
    # Strategy 1: Extract \left(...\right) with balanced parentheses
    # Find all \left( and \right) pairs
    left_pattern = r'\\left\s*\('
    right_pattern = r'\\right\)'
    left_matches = list(re.finditer(left_pattern, text, re.IGNORECASE))
    right_matches = list(re.finditer(right_pattern, text, re.IGNORECASE))
    
    if left_matches and right_matches:
        # Find the last matching pair
        for left_match in reversed(left_matches):
            left_end = left_match.end()
            for right_match in reversed(right_matches):
                if right_match.start() > left_end:
                    # Extract content between \left( and \right)
                    content = text[left_end:right_match.start()].strip()
                    candidate = _clean_candidate(content)
                    if candidate:
                        # Return the full expression including \left( and \right)
                        return text[left_match.start():right_match.end()].strip()
    
    # Strategy 2: Coordinate pattern
    coord_match = _COORDINATE_PATTERN.search(text)
    if coord_match:
        candidate = _clean_candidate(coord_match.group(1))
        if candidate:
            return candidate
    
    # Strategy 3: Answer phrases
    answer_matches = _ANSWER_PHRASE_PATTERN.findall(text)
    if answer_matches:
        candidate = _clean_candidate(re.sub(r"^(?:is|are|equals?|=\s*)", "", 
                                           answer_matches[-1], flags=re.IGNORECASE))
        if candidate:
            return candidate
    
    # Strategy 4: LaTeX math expressions
    latex_matches = _LATEX_MATH_PATTERN.findall(text)
    if latex_matches:
        candidate = _clean_candidate(latex_matches[-1])
        if candidate:
            return candidate
    
    # Strategy 5: Equals pattern
    equals_matches = _EQUALS_PATTERN.findall(text)
    if equals_matches:
        candidate = _clean_candidate(equals_matches[-1])
        if candidate:
            return candidate
    
    # Strategy 6: If text is a pure LaTeX expression (starts with \ and contains LaTeX commands)
    text_stripped = text.strip()
    if text_stripped.startswith('\\') and any(cmd in text_stripped for cmd in ['\\left', '\\frac', '\\boxed', '\\sqrt', '\\pi', '\\alpha', '\\beta']):
        candidate = _clean_candidate(text_stripped)
        if candidate:
            return candidate
    
    # Last resort: check last line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines and re.search(r'[\(\)\[\]\d\\]', lines[-1]):
        last_line = lines[-1]
        for pattern in [r'\(([^\)]+)\)', r'=\s*([^\n]+)']:
            match = re.search(pattern, last_line)
            if match:
                candidate = _clean_candidate(match.group(1))
                if candidate:
                    return candidate
        if len(last_line) < 200:
            return last_line
    
    return None


# ============================================================================
# PRM800K Math Normalize Functions (from math_normalize.py)
# ============================================================================

def _fix_fracs(string: str) -> str:
    """Fix LaTeX fraction formatting."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    """Convert a/b to \\frac{a}{b}."""
    parts = string.split("/")
    if len(parts) != 2:
        return string
    try:
        a, b = int(parts[0]), int(parts[1])
        if string == f"{a}/{b}":
            return f"\\frac{{{a}}}{{{b}}}"
    except:
        pass
    return string


def _remove_right_units(string: str) -> str:
    """Remove units from the right side of the string."""
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string: str) -> str:
    """Fix LaTeX sqrt formatting."""
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    """Normalize answer string (MATH dataset style)."""
    # Basic replacements
    replacements = [
        ("\n", ""), ("\\!", ""), ("\\\\", "\\"), ("tfrac", "frac"), ("dfrac", "frac"),
        ("\\left", ""), ("\\right", ""), ("^{\\circ}", ""), ("^\\circ", ""), ("\\$", ""),
        ("\\%", ""), ("\%", ""), (" .", " 0."), ("{.", "{0."), (" ", "")
    ]
    for old, new in replacements:
        string = string.replace(old, new)
    
    string = _remove_right_units(string)
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    
    string = _fix_sqrt(string)
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    return _fix_a_slash_b(string)


def _normalize_answer_mathd(answer: Optional[str]) -> Optional[str]:
    """Normalize answer using MATH dataset logic."""
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


# ============================================================================
# PRM800K Grader Functions (from grader.py)
# ============================================================================

def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    if not SYMPY_AVAILABLE:
        raise ImportError("sympy not available")
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    if not LATEX2TEXT_AVAILABLE:
        raise ImportError("pylatexenc not available")
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")
    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str) -> str:
    """Automatically make a mixed number evalable e.g. 7 3/4 => 7+3/4"""
    p1 = re.compile(r"([0-9]) +([0-9])")
    step = p1.sub(r"\1+\2", step)  # implicit mults
    return step


def _strip_properly_formatted_commas(expr: str) -> str:
    """Strip properly formatted commas from numbers (e.g., 40,000 -> 40000)."""
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub(r"\1\3\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize_prm800k(expr: str) -> Optional[str]:
    """Normalize answer expressions using PRM800K logic."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m:
        expr = m.group("text")

    # Basic replacements
    expr = expr.replace("\\%", "%").replace("\\$", "$").replace("$", "").replace("%", "")
    expr = expr.replace(" or ", " , ").replace(" and ", " , ")
    expr = expr.replace("million", "*10^6").replace("billion", "*10^9").replace("trillion", "*10^12")

    # Remove units
    units = ["degree", "cm", "centimeter", "meter", "mile", "second", "minute",
             "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard"]
    for unit in units:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if expr and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(r",\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    expr = re.sub(r"- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr).replace(" ", "").replace("{", "").replace("}", "").lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def _count_unknown_letters_in_expr(expr: str) -> int:
    """Count unknown letters in expression."""
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def _should_allow_eval(expr: str) -> bool:
    """Check if expression is safe to evaluate with sympy."""
    # we don't want to try parsing unknown text or functions of more than two variables
    if _count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in _BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in _BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def _are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str) -> bool:
    """Check if two expressions are equal using SymPy."""
    if not SYMPY_AVAILABLE:
        return False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if _should_allow_eval(expr):
            return sympy.simplify(_sympy_parse(expr)) == 0
    except:
        pass
    return False


def _split_tuple(expr: str) -> list[str]:
    """Split the elements in a tuple/interval, while handling well-formatted commas in large numbers."""
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in _TUPLE_CHARS
        and expr[-1] in _TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in _TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def _grade_answer_prm800k(given_answer: str, ground_truth: str) -> bool:
    """
    PRM800K grading logic.
    
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer (MATH style)
    OR
    (b) it normalizes to the same string (PRM800K style)
    OR
    (c) sympy can simplify the difference between the expressions to 0
    """
    if given_answer is None:
        return False

    # Step 1: Try MATH dataset normalization
    ground_truth_normalized_mathd = _normalize_answer_mathd(ground_truth)
    given_answer_normalized_mathd = _normalize_answer_mathd(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    # Step 2: Try PRM800K normalization
    ground_truth_normalized = _normalize_prm800k(ground_truth)
    given_normalized = _normalize_prm800k(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    # Step 3: Handle tuples/intervals
    ground_truth_elems = _split_tuple(ground_truth_normalized)
    given_elems = _split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        # Step 4: Compare elements using SymPy if available
        is_correct = True
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match
                is_correct = False
            else:
                # Use SymPy to check mathematical equivalence
                is_correct = _are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


@registry.asset(
    "metrics",
    "math500_accuracy",
    desc="MATH-500 LaTeX answer accuracy with PRM800K grading logic",
    tags=("math", "math500"),
    default_aggregation="mean",
)
class Math500AccuracyMetric(SimpleMetric):
    """Metric for evaluating MATH-500 answers using PRM800K grading logic."""

    value_key = "acc"

    def compute(self, context: MetricContext) -> MetricResult:
        # STEP 1: Resolve config fields.
        prediction_field = self.args.get("prediction_field", "model_output.answer")
        reference_field = self.args.get("reference_field", "sample.references")
        label_field = self.args.get("label_field", "sample.label")

        # STEP 2: Extract prediction and reference
        sample_dict = extract_field(context, "sample", default={})
        prediction_raw = extract_field(context, prediction_field, default="")
        if isinstance(sample_dict, dict) and "predict_result" in sample_dict:
            prediction_raw = get_text_content_of_first_predict_result(sample_dict) or prediction_raw
        
        extracted_prediction = (_extract_boxed_answer(str(prediction_raw)) if prediction_raw 
                               else None) or (_extract_answer_from_text(str(prediction_raw)) 
                                             if prediction_raw else None)
        prediction = extracted_prediction or (str(prediction_raw) if prediction_raw else "")

        references_raw = (extract_field(context, reference_field, default=None) or
                         (get_first_reference(sample_dict) if isinstance(sample_dict, dict) else None) or
                         extract_field(context, label_field, default=""))
        references = ensure_list_of_strings(references_raw)
        
        if not references:
            return MetricResult(
                sample_id=context.sample_id,
                values={self.value_key: 0.0},
                metadata={"prediction": extracted_prediction or prediction, "reference": None,
                         "grading_method": "prm800k_no_reference"},
            )

        # STEP 3: Try PRM800K grading
        grading_method = f"prm800k_{'sympy' if SYMPY_AVAILABLE else 'nosympy'}"
        is_correct = False
        
        try:
            is_correct = any(_grade_answer_prm800k(prediction, ref) for ref in references)
        except Exception as exc:
            logger.warning("PRM800K grader failed, falling back to normalization: {}", exc)
            pred_norm = normalize_text_advanced(prediction, strip=True, collapse_whitespace=True, 
                                                case_sensitive=True) or ""
            refs_norm = [normalize_text_advanced(r, strip=True, collapse_whitespace=True, 
                                                case_sensitive=True) or "" for r in references]
            refs_norm = [r for r in refs_norm if r]
            is_correct = bool(pred_norm and refs_norm and any(pred_norm == r for r in refs_norm))
            grading_method = "normalization_fallback"

        return MetricResult(
            sample_id=context.sample_id,
            values={self.value_key: 1.0 if is_correct else 0.0},
            metadata={
                "prediction": extracted_prediction or prediction,
                "reference": references[0] if references else None,
                "grading_method": grading_method,
            },
        )

        return MetricResult(
            sample_id=context.sample_id,
            values={self.value_key: score},
            metadata=metadata,
        )


__all__ = ["Math500AccuracyMetric"]
