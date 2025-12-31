import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.legacy_vllm_backend import LegacyVLLMBackend


class DummyModel:
    pass


def make_backend(config=None):
    orig_load = LegacyVLLMBackend.load_model
    LegacyVLLMBackend.load_model = lambda self, cfg: DummyModel()
    try:
        backend = LegacyVLLMBackend(config or {"model_path": "repo", "tokenizer_path": "backend_tok"})
    finally:
        LegacyVLLMBackend.load_model = orig_load
    return backend


class LegacyVLLMBackendConflictTests(unittest.TestCase):
    def test_conflict_raises(self):
        backend = make_backend({"model_path": "repo", "tokenizer_path": "backend_tok"})
        payload = {"_tokenizer_path": "dataset_tok"}
        with self.assertRaises(ValueError):
            backend.prepare_inputs(payload)

    def test_no_conflict_dataset_only(self):
        backend = make_backend({"model_path": "repo"})
        payload = {"_tokenizer_path": "dataset_tok", "messages": [{"role": "user", "content": "hi"}]}
        out = backend.prepare_inputs(payload)
        self.assertEqual(out["cache_suffix"], "-chat_template" if backend._tokenizer else out["cache_suffix"])


if __name__ == "__main__":
    unittest.main()
