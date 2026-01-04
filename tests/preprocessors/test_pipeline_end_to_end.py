import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.loaders.loader_utils import apply_preprocess
from gage_eval.assets.datasets.manager import DataManager, DataSource
from gage_eval.config.pipeline_config import DatasetSpec


class PipelineEndToEndTests(unittest.TestCase):
    def test_preprocess_to_datamanager_transport(self):
        spec = DatasetSpec(
            dataset_id="ds_pipe",
            loader="jsonl",
            params={"tokenizer_path": "dummy_tok"},
        )
        records = [
            {
                "id": "r1",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            }
        ]

        processed = list(
            apply_preprocess(
                records,
                spec,
                data_path="/tmp/data.jsonl",
                doc_to_visual=lambda rec: [{"type": "image_url", "image_url": {"url": "img.png"}}],
            )
        )
        self.assertEqual(len(processed), 1)
        sample = processed[0]

        manager = DataManager()
        manager.register_source(
            DataSource(
                dataset_id="ds_pipe",
                records=processed,
                metadata={"path": "/tmp/data.jsonl"},
                validation=None,
            )
        )
        iterated = list(manager.iter_samples("ds_pipe"))
        self.assertEqual(len(iterated), 1)
        emitted = iterated[0]


if __name__ == "__main__":
    unittest.main()
