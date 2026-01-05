import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.loaders.loader_utils import apply_preprocess, resolve_doc_to_callable
from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.registry import registry
from gage_eval.assets.datasets.sample import sample_from_dict

class _BindingPre(BasePreprocessor):
    def to_sample(self, record, **kwargs):
        sample = dict(record)
        # Ensure messages exist so multimodal merging has a stable anchor point.
        sample.setdefault("messages", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        return sample_from_dict(sample)


class LoaderPreprocessBindingTests(unittest.TestCase):
    def test_doc_to_kwargs_passthrough(self):
        name = "binding_pre_dummy"
        registry.register("dataset_preprocessors", name, _BindingPre, desc="dummy binding")
        spec = DatasetSpec(dataset_id="ds_bind", loader="jsonl", params={"preprocess": name})

        calls = {}

        def _doc_to_visual(sample):
            calls["called"] = True
            return [{"type": "image_url", "image_url": {"url": "img.png"}}]

        records = [{"id": "x1"}]
        processed = list(apply_preprocess(records, spec, data_path="/tmp/data.jsonl", doc_to_visual=_doc_to_visual))

        self.assertEqual(len(processed), 1)

if __name__ == "__main__":
    unittest.main()
