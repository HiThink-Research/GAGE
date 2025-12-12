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


class _TraceablePre(BasePreprocessor):
    def to_sample(self, record, **kwargs):
        sample = dict(record)
        sample.setdefault("messages", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        sample.setdefault("choices", [{"label": "A", "value": "foo"}])
        sample.setdefault("inputs", {"prompt": "hi"})
        return sample


class _ErrorPre(BasePreprocessor):
    def to_sample(self, record, **kwargs):
        raise ValueError("boom")


class PreprocessTraceEventTests(unittest.TestCase):
    def test_trace_events_emitted_in_order(self):
        trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="trace-pre"))
        cfg = ObservabilityConfig(enabled=True)
        pre = _TraceablePre()
        sample = {"id": "s1"}

        pre.transform(sample, trace=trace, observability_config=cfg)

        events = trace.events
        names = [e["event"] for e in events]
        self.assertEqual(names, ["preprocess_start", "preprocess_structured", "preprocess_multimodal", "preprocess_done"])
        payload = events[-1]["payload"]
        self.assertEqual(payload["id"], "s1")
        self.assertIn("cost_ms", payload)
        self.assertGreaterEqual(payload["cost_ms"], 0)

    def test_error_event_emitted_on_exception(self):
        trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="trace-pre-error"))
        cfg = ObservabilityConfig(enabled=True)
        pre = _ErrorPre(on_error="skip")
        sample = {"id": "err1"}

        result = pre.transform(sample, trace=trace, observability_config=cfg)

        self.assertIsNone(result)
        names = [e["event"] for e in trace.events]
        self.assertIn("preprocess_error", names)
        error_payloads = [e["payload"] for e in trace.events if e["event"] == "preprocess_error"]
        self.assertEqual(error_payloads[0]["id"], "err1")
        self.assertIn("boom", error_payloads[0]["error"])

    def test_disabled_config_suppresses_events(self):
        trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="trace-pre-off"))
        cfg = ObservabilityConfig(enabled=False)
        pre = _TraceablePre()

        pre.transform({"id": "off"}, trace=trace, observability_config=cfg)

        self.assertFalse(trace.events)


if __name__ == "__main__":
    unittest.main()
