"""Integration tests for MATH-500 preprocessor."""

import sys
from pathlib import Path
import unittest
from dataclasses import is_dataclass

ROOT = Path(__file__).resolve().parents[3] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.math500.math500_preprocessor import (
    Math500Preprocessor,
)
from gage_eval.assets.datasets.sample import (
    Sample,
)


class Math500PreprocessorIntegrationTests(unittest.TestCase):
    def test_math500_preprocessor_transform(self):
        """Test MATH-500 preprocessor transform method."""
        pre = Math500Preprocessor()
        record = {
            "problem": "What is 2+2?",
            "answer": "4",
            "solution": "2+2=4",
            "subject": "Prealgebra",
            "level": 1,
            "unique_id": "test/prealgebra/1.json",
        }

        # transform returns the structured sample
        sample = pre.transform(record)

        # Check that the sample is properly structured
        self.assertIsNotNone(sample)
        self.assertTrue(is_dataclass(sample))
        self.assertIsNotNone(sample.messages)
        self.assertEqual(len(sample.messages), 1)
        self.assertEqual(sample.messages[0].role, "user")
        self.assertEqual(sample.metadata["subject"], "Prealgebra")
        # Check references (answers mapped to references)
        self.assertEqual(len(sample.references), 1)
        self.assertEqual(sample.references[0], "4")
        self.assertEqual(sample.label, "4")
        self.assertIsNotNone(sample.id)
        self.assertIsNotNone(sample.schema_version)

    def test_math500_preprocessor_with_dataset_id(self):
        """Test MATH-500 preprocessor with custom dataset_id."""
        pre = Math500Preprocessor(dataset_id="custom_math500")
        record = {
            "problem": "What is 2+2?",
            "answer": "4",
        }

        sample = pre.transform(record)

        # Check that transform succeeded and sample has expected structure
        self.assertIsNotNone(sample)
        self.assertTrue(is_dataclass(sample))
        self.assertIsNotNone(sample.messages)


if __name__ == "__main__":
    unittest.main()

