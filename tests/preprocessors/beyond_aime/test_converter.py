"""Tests for BeyondAIME preprocessor."""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.beyond_aime.converter import (
    BeyondAIMEPreprocessor,
)
from gage_eval.assets.datasets.sample import Sample
from dataclasses import is_dataclass


class BeyondAIMEPreprocessorTests(unittest.TestCase):
    """Tests for BeyondAIMEPreprocessor."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.preprocessor = BeyondAIMEPreprocessor()

    def test_to_sample_basic(self) -> None:
        """Test basic sample conversion."""
        problem = "What is 2 + 2?"
        answer = "4"
        record = {
            "problem": problem,
            "answer": answer,
        }

        result = self.preprocessor.to_sample(record)

        self.assertIsNotNone(result)
        self.assertTrue(is_dataclass(result))
        self.assertIsInstance(result, Sample)
        # Check that id is a hash of the problem
        expected_id = hashlib.md5(problem.encode("utf-8")).hexdigest()
        self.assertEqual(result.id, expected_id)
        self.assertEqual(result.label, "4")
        self.assertEqual(result.references, ["4"])

    def test_to_sample_with_latex_problem(self) -> None:
        """Test sample conversion with LaTeX in problem."""
        problem = r"Find the value of $x$ in the equation $x^2 - 5x + 6 = 0$."
        record = {
            "problem": problem,
            "answer": "3",
        }

        result = self.preprocessor.to_sample(record)

        # Check that id is a hash of the problem
        expected_id = hashlib.md5(problem.encode("utf-8")).hexdigest()
        self.assertEqual(result.id, expected_id)
        # Check that problem is in the prompt
        self.assertIn(problem, result.messages[0].content[0].text)
        # Check that boxed instruction is added
        self.assertIn(r"\boxed{}", result.messages[0].content[0].text)
        self.assertEqual(result.label, "3")

    def test_to_sample_with_large_integer_answer(self) -> None:
        """Test sample conversion with large integer answer."""
        problem = "Calculate the sum of all integers from 1 to 100."
        answer = "5050"
        record = {
            "problem": problem,
            "answer": answer,
        }

        result = self.preprocessor.to_sample(record)

        self.assertEqual(result.label, "5050")
        self.assertEqual(result.references, ["5050"])

    def test_to_sample_schema_version(self) -> None:
        """Test that schema version is properly set."""
        record = {
            "problem": "Test problem.",
            "answer": "100",
        }

        result = self.preprocessor.to_sample(record)

        self.assertIsNotNone(result.schema_version)
        self.assertIsInstance(result.schema_version, str)

    def test_to_sample_messages_structure(self) -> None:
        """Test that messages are properly structured."""
        record = {
            "problem": "Test message structure.",
            "answer": "7",
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

    def test_to_sample_consistent_hash(self) -> None:
        """Test that the same problem generates the same hash."""
        problem = "What is the value of $\\pi$ to 2 decimal places?"
        record = {
            "problem": problem,
            "answer": "314",
        }

        result1 = self.preprocessor.to_sample(record)
        result2 = self.preprocessor.to_sample(record)

        self.assertEqual(result1.id, result2.id)

    def test_to_sample_different_problems_different_hashes(self) -> None:
        """Test that different problems generate different hashes."""
        record1 = {
            "problem": "What is 1 + 1?",
            "answer": "2",
        }
        record2 = {
            "problem": "What is 2 + 2?",
            "answer": "4",
        }

        result1 = self.preprocessor.to_sample(record1)
        result2 = self.preprocessor.to_sample(record2)

        self.assertNotEqual(result1.id, result2.id)

    def test_to_sample_metadata(self) -> None:
        """Test that metadata is properly set."""
        record = {
            "problem": "Simple problem.",
            "answer": "42",
        }

        result = self.preprocessor.to_sample(record)

        self.assertIsNotNone(result.metadata)
        self.assertIsInstance(result.metadata, dict)


if __name__ == "__main__":
    unittest.main()
