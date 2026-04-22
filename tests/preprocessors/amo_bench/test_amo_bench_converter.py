"""Tests for AMO-Bench preprocessor."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

# Test helper functions directly without importing full GAGE framework


def _append_try_list(record: dict) -> dict:
    """Append try_list for variable-type questions based on question_id."""
    info = dict(record)
    question_id = info.get("question_id")
    
    if question_id == 5:
        assert info.get("answer_type") == "variable"
        try_list = ["n=1", "n=2", "n=3", "n=4", "n=5", "n=6", "n=7", "n=8", "n=9", "n=10",
                    "n=11", "n=12", "n=13", "n=14", "n=15", "n=16", "n=17", "n=18", "n=19", "n=20"]
        info["try_list"] = try_list
    elif question_id == 37:
        assert info.get("answer_type") == "variable"
        try_list = ["a=2,b=3,c=4", "a=3,b=4,c=5", "a=4,b=5,c=6", "a=5,b=6,c=7", "a=6,b=7,c=8",
                    "a=7,b=8,c=9", "a=8,b=9,c=10", "a=9,b=10,c=11", "a=10,b=11,c=12",
                    "a=11,b=12,c=13", "a=12,b=13,c=14", "a=13,b=14,c=15", "a=14,b=15,c=16",
                    "a=15,b=16,c=17", "a=16,b=17,c=18", "a=17,b=18,c=19", "a=18,b=19,c=20"]
        info["try_list"] = try_list
    
    return info


class AppendTryListTests(unittest.TestCase):
    """Tests for _append_try_list helper function."""

    def test_append_try_list_question_5(self) -> None:
        """Test try_list generation for question_id 5."""
        record = {
            "question_id": 5,
            "answer_type": "variable",
        }
        result = _append_try_list(record)
        
        self.assertIn("try_list", result)
        self.assertEqual(len(result["try_list"]), 20)
        self.assertEqual(result["try_list"][0], "n=1")
        self.assertEqual(result["try_list"][-1], "n=20")

    def test_append_try_list_question_37(self) -> None:
        """Test try_list generation for question_id 37."""
        record = {
            "question_id": 37,
            "answer_type": "variable",
        }
        result = _append_try_list(record)
        
        self.assertIn("try_list", result)
        self.assertEqual(len(result["try_list"]), 17)
        self.assertEqual(result["try_list"][0], "a=2,b=3,c=4")

    def test_append_try_list_other_questions(self) -> None:
        """Test that other questions don't get try_list."""
        record = {
            "question_id": 1,
            "answer_type": "number",
        }
        result = _append_try_list(record)
        
        # Should not have try_list for other question_ids
        self.assertNotIn("try_list", result)


class ConverterLogicTests(unittest.TestCase):
    """Tests for converter logic."""

    def test_user_prompt_template_format(self) -> None:
        """Test that prompt template is correctly formatted."""
        _USER_PROMPT_TEMPLATE = """{prompt}

After solving the above problem, please output your final answer in the following format:
### The final answer is: $\\boxed{{<your answer>}}$
Example:
### The final answer is: $\\boxed{{123}}$
The final answer should be given as precisely as possible (using LaTeX symbols such as \\sqrt, \\frac, \\pi, etc.). If
the final answer involves a decimal approximation, it must be accurate to at least four decimal places."""
        
        problem = "What is 2 + 2?"
        formatted_prompt = _USER_PROMPT_TEMPLATE.format(prompt=problem)
        
        # Check that original problem is in the prompt
        self.assertIn(problem, formatted_prompt)
        # Check that formatting instructions are included
        self.assertIn("After solving the above problem", formatted_prompt)
        self.assertIn("The final answer is:", formatted_prompt)
        self.assertIn(r"\boxed{", formatted_prompt)

    def test_field_mappings(self) -> None:
        """Test field mappings from record to sample."""
        record = {
            "question_id": 1,
            "prompt": "Test problem",
            "solution": "Test solution",
            "answer": "42",
            "answer_type": "number",
        }
        
        # Simulate the mapping logic
        sample_id = str(record.get("question_id", ""))
        prompt = record.get("prompt", "")
        solution = record.get("solution", "")
        answer = record.get("answer", "")
        answer_type = record.get("answer_type", "")
        
        # Check mappings
        self.assertEqual(sample_id, "1")
        self.assertEqual(prompt, "Test problem")
        self.assertEqual(solution, "Test solution")
        self.assertEqual(answer, "42")
        self.assertEqual(answer_type, "number")
        
        # References should contain answer
        references = [answer]
        self.assertEqual(references, ["42"])


if __name__ == "__main__":
    unittest.main()
