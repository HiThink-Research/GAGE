import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.mme import MMEPreprocessor
from gage_eval.assets.datasets.sample import (
    Sample,
)
from dataclasses import is_dataclass, asdict
from PIL import Image
import random
import numpy as np

def generate_random_image_v2(width, height):
    # 生成 [height, width, 3] 的随机 uint8 数组
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(data, 'RGB')
    return img

class MMEPreprocessorTests(unittest.TestCase):
    def test_to_sample(self):
        question_id = "code_reasoning/0020.png"
        question = "Is a python code shown in the picture?"
        decoded_image = generate_random_image_v2(32, 32)
        answer = "Yes"
        category = "code_reasoning"
        sample = {
            "question_id": question_id,
            "question": question,
            "decoded_image": decoded_image,
            "answer": answer,
            "category": category,
        }
        pre = MMEPreprocessor()
        
        ret = pre.to_sample(sample)
        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        self.assertIn(question, ret.messages[-1].content[0].text)
        self.assertEqual(len(ret.messages[-1].content), 2)
        self.assertIsNotNone(ret.id)
        self.assertIsNotNone(ret.schema_version)
        self.assertIsNotNone(ret.references)
        self.assertIsNotNone(ret.label)
        self.assertEqual(ret.references[0], "Yes")
        self.assertEqual(ret.label, "Yes")
        self.assertEqual(ret.metadata["category"], category)
        self.assertEqual(ret.metadata["question_id"], question_id)

    def test_to_sample_with_system_prompt(self):
        question_id = "test_001"
        question = "Is this a test?"
        decoded_image = generate_random_image_v2(32, 32)
        answer = "No"
        sample = {
            "question_id": question_id,
            "question": question,
            "decoded_image": decoded_image,
            "answer": answer,
        }
        pre = MMEPreprocessor()
        
        ret = pre.to_sample(sample, system_prompt="You are a helpful assistant.")
        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        self.assertEqual(len(ret.messages), 2)
        self.assertEqual(ret.messages[0].role, "system")
        self.assertEqual(ret.messages[1].role, "user")
        self.assertEqual(ret.references[0], "No")
        self.assertEqual(ret.id, f"{question_id}:q1")  # ID format is question_id:q1

    def test_mme_preprocessor_answer_variations(self):
        """Test MME preprocessor handles various answer formats."""
        pre = MMEPreprocessor()
        
        test_cases = [
            ("yes", "Yes"),
            ("Yes", "Yes"),
            ("no", "No"),
            ("No", "No"),
        ]
        
        for input_answer, expected in test_cases:
            record = {
                "question_id": f"test_{input_answer}",
                "question": "Test question?",
                "answer": input_answer,
                "decoded_image": generate_random_image_v2(32, 32),
            }
            sample = pre.to_sample(record)
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

        sample = pre.to_sample(record)

        self.assertIsNotNone(sample)
        self.assertTrue(is_dataclass(sample))
        self.assertEqual(len(sample.messages), 1)
        self.assertEqual(len(sample.messages[0].content), 1)  # Only text
        self.assertEqual(sample.messages[0].content[0].type, "text")
        self.assertEqual(sample.references[0], "Yes")
        self.assertEqual(sample.label, "Yes")

    def test_mme_preprocessor_multiple_questions_same_id(self):
        """Test MME preprocessor handles multiple questions with same question_id."""
        pre = MMEPreprocessor()
        question_id = "test_001"
        decoded_image = generate_random_image_v2(32, 32)
        
        # First question
        record1 = {
            "question_id": question_id,
            "question": "First question?",
            "answer": "Yes",
            "decoded_image": decoded_image,
        }
        sample1 = pre.to_sample(record1)
        self.assertEqual(sample1.id, f"{question_id}:q1")
        
        # Second question with same question_id
        record2 = {
            "question_id": question_id,
            "question": "Second question?",
            "answer": "No",
            "decoded_image": decoded_image,
        }
        sample2 = pre.to_sample(record2)
        self.assertEqual(sample2.id, f"{question_id}:q2")

    def test_mme_preprocessor_empty_answer(self):
        """Test MME preprocessor handles empty answer."""
        pre = MMEPreprocessor()
        record = {
            "question_id": "test_001",
            "question": "Is this a test?",
            "answer": "",
            "decoded_image": generate_random_image_v2(32, 32),
        }

        sample = pre.to_sample(record)

        self.assertIsNotNone(sample)
        self.assertTrue(is_dataclass(sample))
        # Empty answer should still create a reference (empty string)
        self.assertEqual(len(sample.references), 1)
        self.assertEqual(sample.references[0], "")
        self.assertEqual(sample.label, "")

    def test_mme_preprocessor_missing_category(self):
        """Test MME preprocessor handles missing category field."""
        pre = MMEPreprocessor()
        record = {
            "question_id": "test_001",
            "question": "Is this a test?",
            "answer": "Yes",
            "decoded_image": generate_random_image_v2(32, 32),
        }

        sample = pre.to_sample(record)

        self.assertIsNotNone(sample)
        self.assertTrue(is_dataclass(sample))
        # Category should be empty string if not provided
        self.assertEqual(sample.metadata.get("category", ""), "")


if __name__ == "__main__":
    unittest.main()
