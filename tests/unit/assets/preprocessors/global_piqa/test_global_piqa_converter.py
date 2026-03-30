import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.global_piqa.global_piqa_converter import GlobalPIQAConverter

from gage_eval.assets.datasets.sample import (
    Sample,
)
from dataclasses import is_dataclass, asdict

class GlobalPIQAConverterTests(unittest.TestCase):
    def test_to_sample(self):
        sample = {
            "example_id": "0",
            "prompt": """answer question: """,
            "solution0": "what",
            "solution1": "why",
            "label": 1,
            "language": "english",
            "answer_index": 0,
        }

        pre = GlobalPIQAConverter()
        
        ret = pre.to_sample(sample)
        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        self.assertIsNotNone(ret.metadata)
        self.assertIsNotNone(ret.messages[0].content[0])
        self.assertIn("""answer question: """, ret.messages[0].content[0].text)
        self.assertEqual(ret.id, "0")
        self.assertIsNotNone(ret.schema_version)
        self.assertIsNotNone(ret.references)
        self.assertIsNotNone(ret.label)
        self.assertEqual(ret.label, 'B')