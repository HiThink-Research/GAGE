import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.registry import registry
from gage_eval.role.model.backends.legacy_vllm_backend import LegacyVLLMBackend


class DummyModel:
    pass


class LegacyVLLMBackendImportTests(unittest.TestCase):
    def test_registry_entry_exists(self):
        cls = registry.get("backends", "legacy_vllm")
        self.assertIs(cls, LegacyVLLMBackend)
        meta = registry.entry("backends", "legacy_vllm")
        self.assertEqual(meta.name, "legacy_vllm")

    def test_can_instantiate_with_stub_model(self):
        orig_load = LegacyVLLMBackend.load_model
        LegacyVLLMBackend.load_model = lambda self, cfg: DummyModel()
        try:
            backend = LegacyVLLMBackend({"model_path": "repo"})
        finally:
            LegacyVLLMBackend.load_model = orig_load
        self.assertIsInstance(backend.model, DummyModel)
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        prepared = backend.prepare_inputs(payload)
        self.assertIn("assistant:", prepared["prompt"])
        self.assertEqual(prepared["request_id"].startswith("legacy_vllm_"), True)
        self.assertIn("cache_suffix", prepared)


if __name__ == "__main__":
    unittest.main()
