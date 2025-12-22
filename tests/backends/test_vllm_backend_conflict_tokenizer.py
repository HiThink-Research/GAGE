import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.vllm_backend import VLLMBackend


class DummyModel:
    pass


def make_backend(config=None):
    orig_load = VLLMBackend.load_model
    VLLMBackend.load_model = lambda self, cfg: DummyModel()
    try:
        backend = VLLMBackend(config or {"model_path": "repo", "tokenizer_path": "backend_tok"})
    finally:
        VLLMBackend.load_model = orig_load
    return backend


class VLLMBackendConflictTests(unittest.TestCase):
    def test_conflict_raises(self):
        backend = make_backend({"model_path": "repo", "tokenizer_path": "backend_tok"})
        payload = {"_tokenizer_path": "dataset_tok"}
        with self.assertRaises(ValueError):
            backend.prepare_inputs(payload)

    def test_dataset_only(self):
        backend = make_backend({"model_path": "repo"})
        payload = {"_tokenizer_path": "dataset_tok", "messages": [{"role": "user", "content": "hi"}]}
        out = backend.prepare_inputs(payload)
        self.assertIn("prompt", out)


if __name__ == "__main__":
    unittest.main()
