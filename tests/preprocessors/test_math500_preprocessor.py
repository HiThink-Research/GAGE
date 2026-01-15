import sys
from pathlib import Path
import unittest
from dataclasses import is_dataclass

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.math500.math500_preprocessor import (
    Math500Preprocessor,
)
from gage_eval.assets.datasets.sample import (
    Sample,
)


class Math500PreprocessorTests(unittest.TestCase):
    def test_to_sample(self):
        """Text QA: emits messages, references, metadata."""
        pre = Math500Preprocessor()
        sample = {
            "problem": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.",
            "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
            "solution": "We have that $r = \\sqrt{0^2 + 3^2} = 3.$",
            "subject": "Precalculus",
            "level": 2,
            "unique_id": "test/precalculus/807.json",
        }
        ret = pre.to_sample(sample)

        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        
        # Messages should contain the problem.
        self.assertIsNotNone(ret.messages)
        self.assertEqual(len(ret.messages), 1)
        self.assertEqual(ret.messages[0].role, "user")
        self.assertIn(sample["problem"], ret.messages[0].content[0].text)

        # References and label should be set.
        self.assertIsNotNone(ret.references)
        self.assertEqual(len(ret.references), 1)
        self.assertEqual(ret.references[0], sample["answer"])
        self.assertEqual(ret.label, sample["answer"])

        # Metadata should contain subject, level, unique_id, and solution.
        self.assertIsNotNone(ret.metadata)
        self.assertEqual(ret.metadata["subject"], "Precalculus")
        self.assertEqual(ret.metadata["level"], 2)
        self.assertEqual(ret.metadata["unique_id"], "test/precalculus/807.json")
        self.assertEqual(ret.metadata["solution"], sample["solution"])

        # ID and schema version should be set.
        self.assertIsNotNone(ret.id)
        self.assertIsNotNone(ret.schema_version)

    def test_to_sample_with_system_prompt(self):
        """Test with system prompt."""
        pre = Math500Preprocessor()
        sample = {
            "problem": "What is 2+2?",
            "answer": "4",
            "subject": "Prealgebra",
            "level": 1,
        }

        ret = pre.to_sample(sample, system_prompt="You are a math assistant.")
        
        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        self.assertEqual(len(ret.messages), 2)  # system + user
        self.assertEqual(ret.messages[0].role, "system")
        self.assertEqual(ret.messages[1].role, "user")

    def test_math500_preprocess_handles_missing_optional_fields(self):
        """Missing optional fields (subject, level, unique_id) are handled gracefully."""
        pre = Math500Preprocessor()
        sample = {
            "problem": "What is 2+2?",
            "answer": "4",
        }
        ret = pre.to_sample(sample)

        # Should handle missing fields gracefully.
        self.assertEqual(ret.metadata["subject"], "")
        self.assertIsNone(ret.metadata["level"])
        self.assertIsNone(ret.metadata["unique_id"])
        self.assertEqual(len(ret.references), 1)
        self.assertEqual(ret.references[0], "4")

    def test_math500_preprocess_handles_empty_answer(self):
        """Empty answer results in empty references list."""
        pre = Math500Preprocessor()
        sample = {
            "problem": "What is 2+2?",
            "answer": "",
        }
        ret = pre.to_sample(sample)

        # Empty answer should result in empty references list.
        self.assertEqual(len(ret.references), 0)
        self.assertIsNone(ret.label)


if __name__ == "__main__":
    unittest.main()

