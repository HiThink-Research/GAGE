import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.manager import DataManager, DataSource


class DocToVisualWithMessagesTests(unittest.TestCase):
    def test_doc_to_visual_runs_even_with_messages(self):
        dm = DataManager()
        source = DataSource(
            dataset_id="d1",
            records=[
                {
                    "id": "s1",
                    "messages": [
                        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "img.png"}}]},
                    ],
                }
            ],
            doc_to_visual=lambda s: ["img.png"],
        )
        dm.register_source(source)
        sample = next(dm.iter_samples("d1"))
        # NOTE: DataManager only passes through here; it does not auto-construct
        # inputs/multi_modal_data.
        self.assertNotIn("inputs", sample)
        self.assertEqual(sample["messages"][0]["content"][0]["image_url"]["url"], "img.png")


if __name__ == "__main__":
    unittest.main()
