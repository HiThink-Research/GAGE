"""Tests for AIME 2026 preprocessor."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.aime.aime2026 import (
    AIME2026Preprocessor,
)
from gage_eval.assets.datasets.sample import Sample
from dataclasses import is_dataclass


class AIME2026PreprocessorTests(unittest.TestCase):
    """Tests for AIME2026Preprocessor."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.preprocessor = AIME2026Preprocessor()

    def test_to_sample_basic(self) -> None:
        """Test basic sample conversion."""
        record = {
            "id": "aime26_001",
            "problem": "What is 2 + 2?",
            "answer": "4",
        }

        result = self.preprocessor.to_sample(record)

        self.assertIsNotNone(result)
        self.assertTrue(is_dataclass(result))
        self.assertIsInstance(result, Sample)
        self.assertEqual(result.id, "aime26_001")
        self.assertEqual(result.label, "4")
        self.assertEqual(result.references, ["4"])

    def test_to_sample_with_latex_problem(self) -> None:
        """Test sample conversion with LaTeX in problem."""
        problem = r"Find the value of $x$ in $x^2 - 5x + 6 = 0$."
        record = {
            "id": "aime26_002",
            "problem": problem,
            "answer": "42",
        }

        result = self.preprocessor.to_sample(record)

        self.assertEqual(result.id, "aime26_002")
        self.assertIn(problem, result.messages[0].content[0].text)
        self.assertIn("ANSWER: $ANSWER", result.messages[0].content[0].text)
        self.assertEqual(result.label, "42")

    def test_to_sample_messages_structure(self) -> None:
        """Test that messages are properly structured."""
        record = {
            "id": "aime26_003",
            "problem": "Test message structure.",
            "answer": "7",
        }

        result = self.preprocessor.to_sample(record)

        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0].role, "user")
        self.assertIsNotNone(result.messages[0].content)
        self.assertGreater(len(result.messages[0].content), 0)
        self.assertEqual(result.messages[0].content[0].type, "text")

    def test_to_sample_schema_version(self) -> None:
        """Test that schema version is properly set."""
        record = {
            "id": "aime26_004",
            "problem": "Test problem.",
            "answer": "100",
        }

        result = self.preprocessor.to_sample(record)

        self.assertIsNotNone(result.schema_version)
        self.assertIsInstance(result.schema_version, str)


if __name__ == "__main__":
    unittest.main()
