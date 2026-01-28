import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.biz_fin_bench_v2.biz_fin_bench_v2_converter import BizFinBenchV2Converter

from gage_eval.assets.datasets.sample import (
    Sample,
)
from dataclasses import is_dataclass, asdict

class BizFinBenchV2ConverterTests(unittest.TestCase):
    def test_to_sample(self):
        sample = {
            "messages":[
                {
                    "content":[
                        {
                            "text": "are you ok?"
                        }
                    ]
                }
            ],
            "choices": [
                {
                    'message':{
                        "content":[
                            {
                                "text": "I am fine!" 
                            }
                        ]
                    }
                }
            ]
        }

        pre = BizFinBenchV2Converter()
        
        ret = pre.to_sample(sample)
        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        self.assertIsNone(ret.metadata)
        self.assertIsNotNone(ret.messages[0].content[0])
        self.assertIn("""are you""", ret.messages[0].content[0].text)
        self.assertIsNotNone(ret.id)
        self.assertIsNotNone(ret.schema_version)
        self.assertIsNotNone(ret.references)