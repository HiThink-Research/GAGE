import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.mrcr.mrcr_converter import MRCRConverter
from gage_eval.assets.datasets.sample import (
    Sample,
)
from dataclasses import is_dataclass, asdict

class MRCRConverterTest(unittest.TestCase):
    def test_to_sample(self):
        sample = {
            "prompt":  '[{"role": "user", "content": "are you ok?"}]',
            "answer": "mWEa9DrPT3**Verse 1** \nIn a world so vast",
            "random_string_to_prepend": "mWEa9DrPT3",
            "n_needles": 2,
            "desired_msg_index": 721,
            "total_messages": 772,
            "n_chars": 708925,
            "date_added": "04-12-2025"
        }
        pre = MRCRConverter()
        ret = pre.to_sample(sample)
        self.assertIsNotNone(ret)
        self.assertIsNotNone(ret.id)
        self.assertTrue(is_dataclass(ret))
        self.assertIsNotNone(ret.metadata)
        self.assertIn("are you", ret.messages[0].content[0].text)
        self.assertIsNotNone(ret.schema_version)
        self.assertIsNotNone(ret.references)