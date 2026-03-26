import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.aime.aime2025 import AIME2025Preprocessor
from gage_eval.assets.datasets.sample import (
    Sample,
)
from dataclasses import is_dataclass, asdict

class AIME2025PreprocessorTests(unittest.TestCase):
    def test_to_sample(self):
        ID = '0'
        Problem = """Let $x,y$ and $z$ be positive real numbers"""

        Answer = '33'
        sample = {
            "id": ID,
            "problem": Problem,
            "answer": Answer
        }
        pre = AIME2025Preprocessor()
        
        ret = pre.to_sample(sample)
        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        self.assertIn(Problem, ret.messages[0].content[0].text)
        self.assertEqual(ret.id, ID)
        self.assertIsNotNone(ret.schema_version)
        self.assertIsNotNone(ret.references)
        self.assertIsNotNone(ret.label)
        self.assertEqual(ret.label, Answer)