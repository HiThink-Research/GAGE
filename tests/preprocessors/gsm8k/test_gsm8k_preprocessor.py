"""Tests for GSM8K preprocessor."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.gsm8k.converter import (
    GSM8KPreprocessor,
)
from gage_eval.assets.datasets.sample import Sample
from dataclasses import is_dataclass


class GSM8KPreprocessorTests(unittest.TestCase):
    """Tests for GSM8KPreprocessor."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.preprocessor = GSM8KPreprocessor()

    def test_to_sample_basic(self) -> None:
        """Test basic sample conversion."""
        record = {
            "question": "What is 2 + 2?",
            "answer": "2 + 2 = 4\n#### 4",
        }

        result = self.preprocessor.to_sample(record)

        self.assertIsNotNone(result)
        self.assertTrue(is_dataclass(result))
        self.assertIsInstance(result, Sample)
        self.assertEqual(result.label, "4")
        self.assertEqual(result.references, ["4"])
        self.assertEqual(result.metadata["cot"], "2 + 2 = 4")

    def test_to_sample_hash_id(self) -> None:
        """Test that sample_id is derived from question hash."""
        record = {
            "question": "What is 2 + 2?",
            "answer": "#### 4",
        }

        result = self.preprocessor.to_sample(record)

        self.assertIsNotNone(result.id)
        self.assertEqual(len(result.id), 16)

    def test_to_sample_multiline_cot(self) -> None:
        """Test sample conversion with multi-line CoT."""
        record = {
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "answer": "It takes 2/2=1 bolt of white fiber\nSo the total amount of fabric is 2+1=3 bolts of fabric\n#### 3",
        }

        result = self.preprocessor.to_sample(record)

        self.assertEqual(result.label, "3")
        self.assertEqual(result.references, ["3"])
        self.assertIn("It takes 2/2=1 bolt of white fiber", result.metadata["cot"])
        self.assertNotIn("####", result.metadata["cot"])

    def test_to_sample_messages_structure(self) -> None:
        """Test that messages are properly structured."""
        record = {
            "question": "Test message structure.",
            "answer": "#### 7",
        }

        result = self.preprocessor.to_sample(record)

        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0].role, "user")
        self.assertIsNotNone(result.messages[0].content)
        self.assertGreater(len(result.messages[0].content), 0)
        self.assertEqual(result.messages[0].content[0].type, "text")
        self.assertIn("ANSWER: $ANSWER", result.messages[0].content[0].text)

    def test_to_sample_schema_version(self) -> None:
        """Test that schema version is properly set."""
        record = {
            "question": "Test problem.",
            "answer": "#### 100",
        }

        result = self.preprocessor.to_sample(record)

        self.assertIsNotNone(result.schema_version)
        self.assertIsInstance(result.schema_version, str)


if __name__ == "__main__":
    unittest.main()
