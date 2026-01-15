"""Integration tests for MME preprocessor."""

import sys
from pathlib import Path
import unittest
from dataclasses import is_dataclass
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[4] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.mme_preprocessor import MMEPreprocessor
from gage_eval.assets.datasets.sample import Sample


def generate_random_image(width, height):
    """Generate a random PIL Image for testing."""
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(data, 'RGB')


class MMEPreprocessorIntegrationTests(unittest.TestCase):
    """Integration tests for MME preprocessor."""

    def test_mme_preprocessor_transform(self):
        """Test MME preprocessor transform method."""
        pre = MMEPreprocessor()
        record = {
            "question_id": "code_reasoning/0020.png",
            "question": "Is a python code shown in the picture? Please answer yes or no.",
            "answer": "Yes",
            "category": "code_reasoning",
            "decoded_image": generate_random_image(32, 32),
        }

        # transform returns the structured sample
        sample = pre.transform(record)

        # Check that the sample is properly structured
        self.assertIsNotNone(sample)
        self.assertTrue(is_dataclass(sample))
        self.assertIsNotNone(sample.messages)
        self.assertEqual(len(sample.messages), 1)
        self.assertEqual(sample.messages[0].role, "user")
        self.assertEqual(len(sample.messages[0].content), 2)  # Text + image
        self.assertEqual(sample.messages[0].content[0].type, "text")
        self.assertEqual(sample.messages[0].content[1].type, "image_url")
        
        # Check references (answers mapped to references)
        self.assertEqual(len(sample.references), 1)
        self.assertEqual(sample.references[0], "Yes")
        self.assertEqual(sample.label, "Yes")
        self.assertIsNotNone(sample.id)
        self.assertIsNotNone(sample.schema_version)
        
        # Check metadata
        self.assertEqual(sample.metadata["category"], "code_reasoning")
        self.assertEqual(sample.metadata["question_id"], "code_reasoning/0020.png")

    def test_mme_preprocessor_with_system_prompt(self):
        """Test MME preprocessor with system prompt."""
        pre = MMEPreprocessor()
        record = {
            "question_id": "test_001",
            "question": "Is this a test?",
            "answer": "No",
            "category": "test",
            "decoded_image": generate_random_image(32, 32),
        }

        sample = pre.transform(record, system_prompt="You are a helpful assistant.")

        self.assertIsNotNone(sample)
        self.assertTrue(is_dataclass(sample))
        self.assertEqual(len(sample.messages), 2)  # System + user
        self.assertEqual(sample.messages[0].role, "system")
        self.assertEqual(sample.messages[1].role, "user")
        self.assertEqual(sample.references[0], "No")

    def test_mme_preprocessor_answer_variations(self):
        """Test MME preprocessor handles various answer formats."""
        pre = MMEPreprocessor()
        
        test_cases = [
            ("yes", "Yes"),
            ("Yes", "Yes"),
            ("y", "Yes"),
            ("no", "No"),
            ("No", "No"),
            ("n", "No"),
        ]
        
        for input_answer, expected in test_cases:
            record = {
                "question_id": f"test_{input_answer}",
                "question": "Test question?",
                "answer": input_answer,
                "decoded_image": generate_random_image(32, 32),
            }
            sample = pre.transform(record)
            self.assertEqual(sample.references[0], expected, f"Failed for input: {input_answer}")
            self.assertEqual(sample.label, expected)

    def test_mme_preprocessor_no_image(self):
        """Test MME preprocessor without image."""
        pre = MMEPreprocessor()
        record = {
            "question_id": "test_001",
            "question": "Is this a test?",
            "answer": "Yes",
            "category": "test",
        }

        sample = pre.transform(record)

        self.assertIsNotNone(sample)
        self.assertTrue(is_dataclass(sample))
        self.assertEqual(len(sample.messages), 1)
        self.assertEqual(len(sample.messages[0].content), 1)  # Only text
        self.assertEqual(sample.messages[0].content[0].type, "text")

    def test_mme_preprocessor_with_dataset_id(self):
        """Test MME preprocessor with custom dataset_id."""
        pre = MMEPreprocessor()
        record = {
            "question_id": "test_001",
            "question": "Is this a test?",
            "answer": "Yes",
            "decoded_image": generate_random_image(32, 32),
        }

        sample = pre.transform(record, dataset_id="mme_test")

        self.assertIsNotNone(sample)
        self.assertTrue(is_dataclass(sample))
        # Dataset ID should be preserved in metadata
        self.assertIsNotNone(sample.id)


if __name__ == "__main__":
    unittest.main()
