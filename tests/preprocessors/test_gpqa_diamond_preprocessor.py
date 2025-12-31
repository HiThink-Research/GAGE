import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.gpqa.gpqa_diamond_preprocessor import GpqaDiamondPreprocessor
from gage_eval.assets.datasets.sample import (
    Sample,
)
from dataclasses import is_dataclass, asdict

class GpqaDiamondPreprocessorTests(unittest.TestCase):
    def test_to_sample(self):
        question = "Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?"
        correct_answer = "10^-4 eV"
        incorrect_answer_1 = "10^-11 eV"
        incorrect_answer_2 = "10^-8 eV"
        incorrect_answer_3 = "10^-9 eV"
        sample = {
            "Question": question,
            "Correct Answer": correct_answer,
            "Incorrect Answer 1": incorrect_answer_1,
            "Incorrect Answer 2": incorrect_answer_2,
            "Incorrect Answer 3": incorrect_answer_3
        }
        pre = GpqaDiamondPreprocessor()
        
        ret = pre.to_sample(sample, gpqa_prompt_type='zero_shot')
        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        self.assertIsNotNone(ret.options)
        self.assertEqual(len(ret.options), 4)
        self.assertIn(question, ret.messages[0].content[0].text)
        self.assertIsNotNone(ret.id)
        self.assertIsNotNone(ret.schema_version)
        self.assertIsNotNone(ret.references)
        self.assertIsNotNone(ret.label)