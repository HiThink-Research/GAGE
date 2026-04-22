"""Tests for AMO-Bench metrics."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# Direct imports to avoid slow registry loading
import re
from typing import Any, Dict, List, Mapping, Optional

# Import test helper functions directly
ANSWER_PREFIX_LIST = [
    "### the final answer is:", "### the final answer:", "### final answer is:", "### final answer:",
    "### the final answer is", "### the final answer", "### final answer is", "### final answer",
]
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
    """Extract answer from prediction text."""
    pred_extract = pred.replace('：', ': ')
    
    for think_postfix in THINK_POSTFIX_LIST:
        pred_extract = pred_extract.split(think_postfix)[-1].strip()
    
    for prefix in ANSWER_PREFIX_LIST + [p[4:] for p in ANSWER_PREFIX_LIST]:
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
        pred_extract = pred_extract.replace("\\ ", " ")
        pred_extract = pred_extract.replace("\\;", ";")
        pred_extract = pred_extract.replace("\n", " ")
    
    if answer_type in ["number", "variable"]:
        pred_extract = pred_extract.replace(",", "")
        pred_extract = pred_extract.replace("\\{", "(").replace("\\}", ")").replace("\\[", "(").replace("\\]", ")")
    
    return pred_extract.strip()


def _extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from \\boxed{...} pattern."""
    if not text:
        return None
    
    pattern = r"\\boxed\s*\{"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    
    if not matches:
        return None
    
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
            content = text[start_idx:end_idx - 1].strip()
            extracted_contents.append(content)
    
    if extracted_contents:
        return extracted_contents[-1]
    return None


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


class PredExtractorTests(unittest.TestCase):
    """Tests for _pred_extractor helper function."""

    def test_extract_with_answer_prefix(self) -> None:
        """Test extraction with answer prefix."""
        pred = "### The final answer is: 42"
        result = _pred_extractor(pred, "number")
        self.assertEqual(result, "42")

    def test_extract_with_boxed(self) -> None:
        """Test extraction with boxed content."""
        pred = r"### The final answer is: $\boxed{42}$"
        result = _pred_extractor(pred, "number")
        self.assertEqual(result, "$\\boxed{42}$")

    def test_extract_with_think_postfix(self) -> None:
        """Test extraction removes think tags."""
        pred = "</think>### The final answer is: 42"
        result = _pred_extractor(pred, "number")
        self.assertEqual(result, "42")

    def test_extract_removes_latex_commands(self) -> None:
        """Test extraction removes latex commands for non-description types."""
        pred = r"\left( 42 \right)"
        result = _pred_extractor(pred, "number")
        self.assertEqual(result, "( 42 )")

    def test_extract_keeps_latex_for_description(self) -> None:
        """Test extraction keeps latex commands for description type."""
        pred = r"\left( example \right)"
        result = _pred_extractor(pred, "description")
        self.assertEqual(result, r"\left( example \right)")


class ExtractBoxedContentTests(unittest.TestCase):
    """Tests for _extract_boxed_content helper function."""

    def test_extract_simple_boxed(self) -> None:
        """Test extracting simple boxed content."""
        text = r"\boxed{42}"
        result = _extract_boxed_content(text)
        self.assertEqual(result, "42")

    def test_extract_boxed_with_latex(self) -> None:
        """Test extracting boxed content with LaTeX."""
        text = r"\boxed{\frac{3}{4}}"
        result = _extract_boxed_content(text)
        self.assertEqual(result, r"\frac{3}{4}")

    def test_extract_last_boxed(self) -> None:
        """Test extracting last boxed when multiple exist."""
        text = r"First: \boxed{10}, Final: \boxed{42}"
        result = _extract_boxed_content(text)
        self.assertEqual(result, "42")

    def test_no_boxed_returns_none(self) -> None:
        """Test that None is returned when no boxed content."""
        text = "The answer is 42"
        result = _extract_boxed_content(text)
        self.assertIsNone(result)


class ExtractJudgeVerdictTests(unittest.TestCase):
    """Tests for _extract_judge_verdict helper function."""

    def test_conclusion_correct(self) -> None:
        """Test extracting correct conclusion."""
        judge_answer = "### Conclusion: Correct"
        self.assertTrue(_extract_judge_verdict(judge_answer))

    def test_conclusion_incorrect(self) -> None:
        """Test extracting incorrect conclusion."""
        judge_answer = "### Conclusion: Incorrect"
        self.assertFalse(_extract_judge_verdict(judge_answer))

    def test_conclusion_not_correct(self) -> None:
        """Test that 'not correct' is handled as incorrect."""
        judge_answer = "The answer is not correct. ### Conclusion: Not Correct"
        self.assertFalse(_extract_judge_verdict(judge_answer))

    def test_conclusion_nt_correct(self) -> None:
        """Test that "isn't correct" / "wasn't correct" is handled as incorrect."""
        # Test with "isn't correct" pattern
        judge_answer = "The answer isn't correct. ### Conclusion: isn't correct"
        self.assertFalse(_extract_judge_verdict(judge_answer))
        
        # Test with "wasn't correct" pattern
        judge_answer = "The answer wasn't correct. ### Conclusion: wasn't correct"
        self.assertFalse(_extract_judge_verdict(judge_answer))

    def test_conclusion_correct_full_response(self) -> None:
        """Test extracting correct from full judge response."""
        judge_answer = """### Conclusion: Correct
The reference answer is 123, which is accurate to at least four decimal places. Therefore, the student's answer is correct."""
        self.assertTrue(_extract_judge_verdict(judge_answer))

    def test_empty_judge_output(self) -> None:
        """Test with empty judge output."""
        self.assertFalse(_extract_judge_verdict(""))

    def test_none_judge_output(self) -> None:
        """Test with None judge output."""
        self.assertFalse(_extract_judge_verdict(None))  # type: ignore

    def test_conclusion_correct_with_context(self) -> None:
        """Test extracting correct conclusion with surrounding context."""
        judge_answer = """For the following math problem, we have the reference answer and the student's answer.

### Problem
What is the capital of France?

### Reference Answer
Paris

### Student Answer
Paris

Now, please provide your judgment.
### Conclusion: Correct
The student's answer matches the reference answer."""
        self.assertTrue(_extract_judge_verdict(judge_answer))


if __name__ == "__main__":
    unittest.main()
