import sys
from pathlib import Path
import unittest
from dataclasses import is_dataclass

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.charxiv import CharXivReasoningPreprocessor  # noqa: E402
from gage_eval.assets.datasets.sample import Sample  # noqa: E402


def generate_random_image(width: int, height: int) -> Image.Image:
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(data, "RGB")


class CharXivReasoningPreprocessorTests(unittest.TestCase):
    def test_to_sample_basic(self):
        img = generate_random_image(64, 64)
        record = {
            "image": img,
            "category": "cs",
            "year": "2020",
            "original_figure_path": "foo/bar.jpg",
            "original_id": "2004.03856",
            "figure_path": "images/0.jpg",
            "num_subplots": 1,
            "subplot_row": 0,
            "subplot_col": 0,
            "subplot_loc": None,
            "reasoning_q": "Which model has the highest accuracy?",
            "reasoning_q_source": 1,
            "reasoning_a": "Joint-CNN",
            "reasoning_a_type": 1,
        }

        pre = CharXivReasoningPreprocessor()
        sample = pre.to_sample(record)

        self.assertIsNotNone(sample)
        self.assertTrue(is_dataclass(sample))
        self.assertIsInstance(sample, Sample)

        # messages
        self.assertEqual(sample.messages[-1].role, "user")
        self.assertIn("Which model has the highest accuracy?", sample.messages[-1].content[0].text)

        # references / label
        self.assertEqual(sample.references[0], "Joint-CNN")
        self.assertEqual(sample.label, "Joint-CNN")

        # metadata passthrough
        self.assertEqual(sample.metadata["category"], "cs")
        self.assertEqual(sample.metadata["year"], "2020")
        self.assertEqual(sample.metadata["original_id"], "2004.03856")
        self.assertEqual(sample.metadata["figure_path"], "images/0.jpg")
        self.assertEqual(sample.metadata["num_subplots"], 1)
        self.assertEqual(sample.metadata["subplot_row"], 0)
        self.assertEqual(sample.metadata["subplot_col"], 0)

        # id should be stable given original_id + subplot indices
        self.assertEqual(sample.id, "2004.03856_r0_c0")


if __name__ == "__main__":
    unittest.main()

