import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.manager import DataManager, DataSource


class DocToVisualWithInputsPromptTests(unittest.TestCase):
    def test_doc_to_visual_runs_when_inputs_prompt_exists(self):
        dm = DataManager()
        source = DataSource(
            dataset_id="d1",
            records=[
                {
                    "id": "s1",
                    "inputs": {"prompt": "hi"},
                    "messages": [
                        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "img.png"}}]},
                    ],
                }
            ],
            doc_to_visual=lambda s: ["img.png"],
        )
        dm.register_source(source)
        sample = next(dm.iter_samples("d1"))
        # NOTE: DataManager only passes through here; it does not auto-merge doc_to_visual.
        # This ensures `inputs` stays unchanged.
        self.assertIn("inputs", sample)
        self.assertEqual(sample["inputs"]["prompt"], "hi")
        self.assertNotIn("multi_modal_data", sample["inputs"])


if __name__ == "__main__":
    unittest.main()
