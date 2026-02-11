import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.mmsu.mmsu_converter import MMSUConverter

from gage_eval.assets.datasets.sample import (
    Sample,
)
from dataclasses import is_dataclass, asdict

class MMSUConverterTests(unittest.TestCase):
    def test_to_sample(self):
        sample = {"id": "code_switch_question_answering_7affd9a4-03cf-4eee-a3be-e4580a8df238", "task_name": "code_switch_question_answering", "audio_path": "/audio/code_switch_question_answering_7affd9a4-03cf-4eee-a3be-e4580a8df238.wav", "question": "What did the speaker receive from their friend?", "choice_a": "A cake.", "choice_b": "Chocolate.", "choice_c": "A fruit basket.", "choice_d": "A drink.", "answer_gt": "Chocolate.", "category": "Reasoning", "sub-category": "Linguistics", "sub-sub-category": "Semantics", "linguistics_sub_discipline": "Semantics"}
        pre = MMSUConverter()
        ret = pre.to_sample(sample,
                            audio_path_root="/mnt/aime_data_ssd/user_workspace/zhuwenqiao/GAGE_workspace/data/mmsu")
        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        self.assertIsNotNone(ret.metadata)
        self.assertIsNotNone(ret.messages[0].content[0])
        self.assertIn("""What did the speaker receive""", ret.messages[0].content[1].text)
        self.assertEqual(len(ret.messages[0].content), 2)
        self.assertIsNotNone(ret.schema_version)
        self.assertIsNotNone(ret.references)
        self.assertIsNotNone(ret.label)
        self.assertEqual(ret.label, 'B')
