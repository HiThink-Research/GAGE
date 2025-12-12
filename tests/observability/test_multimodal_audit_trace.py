import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.observability.config import ObservabilityConfig
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder


class _VisualPre(BasePreprocessor):
    def to_sample(self, record, **kwargs):
        sample = dict(record)
        sample.setdefault("messages", [{"role": "user", "content": [{"type": "text", "text": "see"}]}])
        sample.setdefault("inputs", {"prompt": "see this"})
        sample.setdefault("choices", [])
        return sample


class PreprocessMultimodalAuditTests(unittest.TestCase):
    def test_multimodal_event_counts_images(self):
        pre = _VisualPre()
        trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="mm-audit"))
        cfg = ObservabilityConfig(enabled=True)

        def _doc_to_visual(sample):
            return [{"type": "image_url", "image_url": {"url": "img.png"}}]

        pre.transform(
            {"id": "mm1"},
            trace=trace,
            observability_config=cfg,
            doc_to_visual=_doc_to_visual,
        )

        events = [event for event in trace.events if event["event"] == "preprocess_multimodal"]
        self.assertEqual(len(events), 1)
        payload = events[0]["payload"]
        self.assertEqual(payload["images"], 1)
        self.assertEqual(payload["audios"], 0)
        self.assertEqual(payload["videos"], 0)
        self.assertEqual(payload["files"], 0)


if __name__ == "__main__":
    unittest.main()
