import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.mmlu_pro.mmlu_pro_converter import MMLUProConverter
from gage_eval.assets.datasets.sample import (
    Sample,
)
from dataclasses import is_dataclass, asdict

class MMLUProConverterTest(unittest.TestCase):
    def test_to_sample(self):
        sample = {
            "question_id": "0",
            "question": """The symmetric group $S_n$ has $
\factorial{n}$ elements, hence it is not true that $S_{10}$ has 10 elements.
Find the characteristic of the ring 2Z.""",
            "options": ["0","30","3","10","12","50","2","100","20","5"],
            "answer": 'A',
            "answer_index": 0,
            "cot_content": """A: Let's think step by step. """,
            "category": "math",
            "src": "cot_lib-abstract_algebra"
        }        

        pre = MMLUProConverter()
        id = "0"
        answer = 'A'
        question = "The symmetric group"
        ret = pre.to_sample(sample)
        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        self.assertIsNotNone(ret.metadata)
        self.assertIn(question, ret.messages[0].content[0].text)
        #self.assertEqual(len(ret.messages), 1)
        self.assertEqual(ret.id, id)
        self.assertIsNotNone(ret.schema_version)
        self.assertIsNotNone(ret.references)
        self.assertIsNotNone(ret.label)
        self.assertEqual(ret.label, answer)
        
