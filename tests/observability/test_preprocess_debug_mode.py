import json
import os
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.observability.config import ObservabilityConfig
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder


class _ErrorPre(BasePreprocessor):
    def to_sample(self, record, **kwargs):
        raise ValueError("boom")


class _OKPre(BasePreprocessor):
    def to_sample(self, record, **kwargs):
        sample = dict(record)
        sample.setdefault("messages", [{"role": "user", "content": [{"type": "text", "text": "hello"}]}])
        sample.setdefault("inputs", {"prompt": "hello"})
        sample.setdefault("choices", [])
        return sample


class PreprocessDebugModeTests(unittest.TestCase):
    def test_debug_forces_raise_and_logs_error(self):
        pre = _ErrorPre(on_error="skip")
        trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="debug-pre"))
        cfg = ObservabilityConfig(enabled=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "debug.jsonl"
            os.environ["GAGE_EVAL_DEBUG_PREPROCESS"] = "1"
            os.environ["GAGE_EVAL_DEBUG_PREPROCESS_DIR"] = str(log_file)
            os.environ["GAGE_EVAL_DEBUG_PREPROCESS_SAMPLE"] = "1"
            with self.assertRaises(ValueError):
                pre.transform({"id": "dbg"}, trace=trace, observability_config=cfg)
            self.assertTrue(log_file.exists())
            content = log_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertTrue(content)
            payload = json.loads(content[-1])
            self.assertEqual(payload["stage"], "error")
            self.assertEqual(payload["sample_id"], "dbg")
        os.environ.pop("GAGE_EVAL_DEBUG_PREPROCESS", None)
        os.environ.pop("GAGE_EVAL_DEBUG_PREPROCESS_DIR", None)
        os.environ.pop("GAGE_EVAL_DEBUG_PREPROCESS_SAMPLE", None)

    def test_debug_logs_diff_on_success(self):
        pre = _OKPre()
        trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="debug-pre-ok"))
        cfg = ObservabilityConfig(enabled=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "debug.jsonl"
            os.environ["GAGE_EVAL_DEBUG_PREPROCESS"] = "1"
            os.environ["GAGE_EVAL_DEBUG_PREPROCESS_DIR"] = str(log_file)
            os.environ["GAGE_EVAL_DEBUG_PREPROCESS_SAMPLE"] = "1"
            pre.transform({"id": "ok"}, trace=trace, observability_config=cfg)
            content = log_file.read_text(encoding="utf-8").strip().splitlines()
            payload = json.loads(content[-1])
            self.assertEqual(payload["stage"], "done")
            self.assertEqual(payload["sample_id"], "ok")
            self.assertIn("post", payload)
        os.environ.pop("GAGE_EVAL_DEBUG_PREPROCESS", None)
        os.environ.pop("GAGE_EVAL_DEBUG_PREPROCESS_DIR", None)
        os.environ.pop("GAGE_EVAL_DEBUG_PREPROCESS_SAMPLE", None)


if __name__ == "__main__":
    unittest.main()
