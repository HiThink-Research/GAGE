import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.utils.mapping import map_question_option_answer, resolve_correct_choice
from gage_eval.assets.datasets.utils.multimodal import collect_content_fragments


class FieldMappingTests(unittest.TestCase):
    def test_question_option_answer_mapping(self):
        sample = {
            "question": "What is 1+1?",
            "choices": ["1", "2", "3"],
            "answer": "B",
        }
        messages, choices, meta = map_question_option_answer(sample)
        self.assertEqual(messages[0]["content"][0]["text"], "What is 1+1?")
        self.assertEqual(len(choices), 3)
        self.assertEqual(choices[1]["message"]["content"][0]["text"], "2")
        self.assertEqual(meta["option_map"]["B"], "2")
        self.assertEqual(meta["correct_choice"], "B")

    def test_collect_content_fragments_resolves_root(self):
        sample = {
            "messages": [
                {
                    "content": [
                        {"type": "image_url", "image_url": {"url": "imgs/a.png"}},
                        {"type": "text", "text": "desc"},
                    ]
                }
            ]
        }
        fragments = collect_content_fragments(sample, content_field="messages.0.content", content_root="/data")
        self.assertEqual(fragments[0]["image_url"]["url"], "/data/imgs/a.png")
        self.assertEqual(fragments[1]["text"], "desc")

    def test_resolve_correct_choice_numeric_index(self):
        pairs = [("A", "opt1"), ("B", "opt2"), ("C", "opt3")]
        choice = resolve_correct_choice(2, pairs, answer_index_base=1)
        self.assertEqual(choice, "B")


if __name__ == "__main__":
    unittest.main()
