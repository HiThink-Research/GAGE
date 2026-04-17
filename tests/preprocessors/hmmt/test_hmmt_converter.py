"""Tests for HMMT Feb 2025 preprocessor."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.hmmt.hmmt_converter import (
    HMMTFeb2025Preprocessor,
)
from gage_eval.assets.datasets.sample import Sample
from dataclasses import is_dataclass


class HMMTFeb2025PreprocessorTests(unittest.TestCase):
    """Tests for HMMTFeb2025Preprocessor."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.preprocessor = HMMTFeb2025Preprocessor()

    def test_to_sample_basic(self) -> None:
        """Test basic sample conversion."""
        record = {
            "problem_idx": "problem_001",
            "problem": "What is 2 + 2?",
            "answer": "4",
            "problem_type": ["algebra"],
        }

        result = self.preprocessor.to_sample(record)

        self.assertIsNotNone(result)
        self.assertTrue(is_dataclass(result))
        self.assertIsInstance(result, Sample)
        self.assertEqual(result.id, "problem_001")
        self.assertEqual(result.label, "4")
        self.assertEqual(result.references, ["4"])

    def test_to_sample_with_latex_problem(self) -> None:
        """Test sample conversion with LaTeX in problem."""
        problem = r"Find the value of $x$ in the equation $x^2 - 5x + 6 = 0$."
        record = {
            "problem_idx": "problem_002",
            "problem": problem,
            "answer": "2",
            "problem_type": ["algebra", "quadratic"],
        }

        result = self.preprocessor.to_sample(record)

        self.assertEqual(result.id, "problem_002")
        # Check that problem is in the prompt
        self.assertIn(problem, result.messages[0].content[0].text)
        # Check that boxed instruction is added
        self.assertIn(r"\boxed{}", result.messages[0].content[0].text)
        self.assertEqual(result.label, "2")

    def test_to_sample_with_fraction_answer(self) -> None:
        """Test sample conversion with fraction answer."""
        record = {
            "problem_idx": "problem_003",
            "problem": "What is 1/2 + 1/3?",
            "answer": r"\frac{5}{6}",
            "problem_type": ["arithmetic"],
        }

        result = self.preprocessor.to_sample(record)

        self.assertEqual(result.label, r"\frac{5}{6}")
        self.assertEqual(result.references, [r"\frac{5}{6}"])

    def test_to_sample_with_multiple_problem_types(self) -> None:
        """Test sample conversion with multiple problem types."""
        record = {
            "problem_idx": "problem_004",
            "problem": "A complex geometry problem.",
            "answer": "42",
            "problem_type": ["geometry", "trigonometry", "algebra"],
        }

        result = self.preprocessor.to_sample(record)

        self.assertIsNotNone(result.metadata)
        self.assertIn("problem_type", result.metadata)
        self.assertEqual(
            result.metadata["problem_type"],
            ["geometry", "trigonometry", "algebra"],
        )

    def test_to_sample_empty_problem_type(self) -> None:
        """Test sample conversion with empty problem type."""
        record = {
            "problem_idx": "problem_005",
            "problem": "Simple problem.",
            "answer": "0",
            "problem_type": [],
        }

        result = self.preprocessor.to_sample(record)

        self.assertIsNotNone(result.metadata)
        self.assertEqual(result.metadata["problem_type"], [])

    def test_to_sample_schema_version(self) -> None:
        """Test that schema version is properly set."""
        record = {
            "problem_idx": "problem_006",
            "problem": "Test problem.",
            "answer": "100",
            "problem_type": ["test"],
        }

        result = self.preprocessor.to_sample(record)

        self.assertIsNotNone(result.schema_version)
        self.assertIsInstance(result.schema_version, str)

    def test_to_sample_messages_structure(self) -> None:
        """Test that messages are properly structured."""
        record = {
            "problem_idx": "problem_007",
            "problem": "Test message structure.",
            "answer": "7",
            "problem_type": ["test"],
        }

        result = self.preprocessor.to_sample(record)

        # Should have exactly one message
        self.assertEqual(len(result.messages), 1)
        # Message should be from user
        self.assertEqual(result.messages[0].role, "user")
        # Message should have content
        self.assertIsNotNone(result.messages[0].content)
        self.assertGreater(len(result.messages[0].content), 0)
        # Content should be text type
        self.assertEqual(result.messages[0].content[0].type, "text")


if __name__ == "__main__":
    unittest.main()
