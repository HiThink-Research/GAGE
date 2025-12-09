import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.loaders.loader_utils import apply_preprocess
from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.observability.config import ObservabilityConfig
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.registry import registry
from gage_eval.reporting.recorders import InMemoryRecorder


class _TracingPre(BasePreprocessor):
    def to_sample(self, record, **kwargs):
        sample = dict(record)
        sample.setdefault("messages", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        sample.setdefault("choices", [])
        sample.setdefault("inputs", {"prompt": "hi"})
        return sample


class LoaderTracePassthroughTests(unittest.TestCase):
    def test_trace_emitted_when_enabled(self):
        name = "trace_passthrough_pre"
        registry.register("dataset_preprocessors", name, _TracingPre, desc="trace passthrough dummy")
        spec = DatasetSpec(dataset_id="trace_ds", loader="jsonl", params={"preprocess": name})
        trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="trace-pass"))
        cfg = ObservabilityConfig(enabled=True)

        processed = list(
            apply_preprocess(
                [{"id": "s1"}],
                spec,
                data_path="/tmp/data.jsonl",
                doc_to_text=None,
                doc_to_visual=None,
                doc_to_audio=None,
                trace=trace,
                observability_config=cfg,
            )
        )

        self.assertEqual(len(processed), 1)
        events = [event["event"] for event in trace.events]
        self.assertIn("preprocess_start", events)
        self.assertIn("preprocess_done", events)

    def test_trace_suppressed_when_disabled(self):
        name = "trace_passthrough_disabled_pre"
        registry.register("dataset_preprocessors", name, _TracingPre, desc="trace passthrough disabled")
        spec = DatasetSpec(dataset_id="trace_ds_off", loader="jsonl", params={"preprocess": name})
        trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="trace-off"))
        cfg = ObservabilityConfig(enabled=False)

        list(
            apply_preprocess(
                [{"id": "s2"}],
                spec,
                data_path="/tmp/data.jsonl",
                doc_to_text=None,
                doc_to_visual=None,
                doc_to_audio=None,
                trace=trace,
                observability_config=cfg,
            )
        )

        self.assertFalse(trace.events)


if __name__ == "__main__":
    unittest.main()
