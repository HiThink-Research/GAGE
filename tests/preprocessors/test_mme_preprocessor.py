import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.mme_preprocessor import MMEPreprocessor
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
        self.assertEqual(ret.id, question_id)
