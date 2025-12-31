import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.mathvista.mathvista_chat_preprocessor import MathVistaChatPreprocessor
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

class MathVistaChatPreprocessorTests(unittest.TestCase):
    def test_to_sample(self):
        pid = "1"
        question = "how many dogs in this image?"
        decoded_image =generate_random_image_v2(32, 32)
        answer = "1"
        question_type = "free_form"
        answer_type = "integer"    
        sample = {
            "unit": 'g',
            "image": "fake.jpg",
            "precision": 1,
            "choices": None,
            "query": "test",
            "pid":pid,
            "question": question,
            "decoded_image": decoded_image,
            "answer": answer,
            "question_type": question_type,
            "answer_type": answer_type,
            "caption": None,
            "ocr": None
        }
        pre = MathVistaChatPreprocessor()
        
        ret = pre.to_sample(sample)
        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        self.assertIn(question, ret.messages[0].content[0].text)
        self.assertEqual(len(ret.messages[0].content), 2)
        self.assertIsNotNone(ret.id)
        self.assertIsNotNone(ret.schema_version)
        self.assertIsNotNone(ret.references)
        self.assertIsNotNone(ret.label)

    def test_to_sample_choice(self):
        pid = "1"
        question = "how many dogs in this image?"
        decoded_image =generate_random_image_v2(32, 32)
        answer = "1"
        question_type = "multi_choice"
        answer_type = "text"    
        sample = {
            "unit": 'g',
            "image": "fake.jpg",
            "precision": 1,
            "choices": [1,2,3,4],
            "query": "test",
            "pid":pid,
            "question": question,
            "decoded_image": decoded_image,
            "answer": answer,
            "question_type": question_type,
            "answer_type": answer_type,
            "caption": None,
            "ocr": None
        }
        pre = MathVistaChatPreprocessor()
        
        ret = pre.to_sample(sample)
        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        self.assertIn(question, ret.messages[0].content[0].text)
        self.assertEqual(len(ret.messages[0].content), 2)
        self.assertEqual(len(ret.options), 4)
        self.assertEqual(ret.id, pid)
        self.assertIsNotNone(ret.id)
        self.assertIsNotNone(ret.schema_version)
        self.assertIsNotNone(ret.references)
        self.assertIsNotNone(ret.label)        