import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.legacy_vllm_backend import LegacyVLLMBackend


class DummyEngine:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        rid = kwargs.get("request_id", "")
        text = kwargs.get("prompt") or ""
        return {"outputs": [{"text": f"{text}-resp-{rid}"}]}

    def abort(self, *_args, **_kwargs):
        return None


def make_backend():
    orig_load = LegacyVLLMBackend.load_model
    LegacyVLLMBackend.load_model = lambda self, cfg: DummyEngine()
    try:
        backend = LegacyVLLMBackend({"model_path": "repo"})
    finally:
        LegacyVLLMBackend.load_model = orig_load
    return backend


class LegacyVLLMBackendSmokeTests(unittest.TestCase):
    def test_spawn_env_set(self):
        backend = make_backend()
        import os

        self.assertEqual(os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"), "spawn")
        self.assertIsInstance(backend.model, DummyEngine)

    def test_generate_text_path(self):
        backend = make_backend()
        payload = {"messages": [{"role": "user", "content": "hi"}], "request_id": "req"}
        prepared = backend.prepare_inputs(payload)
        out = backend.generate(prepared)
        self.assertIn("answer", out)
        self.assertIn("req", out["answer"])
        self.assertEqual(len(backend.model.calls), 1)

    def test_generate_sample_n(self):
        backend = make_backend()
        payload = {"prompt": "hello", "request_id": "rid", "sample_n": 2}
        prepared = backend.prepare_inputs(payload)
        out = backend.generate(prepared)
        self.assertEqual(out["_sample_n"], 2)
        self.assertEqual(len(out["answer"]), 2)
        self.assertEqual({c["request_id"] for c in backend.model.calls}, {"rid_0", "rid_1"})


if __name__ == "__main__":
    unittest.main()
