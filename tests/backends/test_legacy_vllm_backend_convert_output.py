import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.legacy_vllm_backend import LegacyVLLMBackend


class DummyModel:
    def generate(self, **kwargs):
        return {"outputs": [{"text": "hello"}]}


def make_backend():
    orig_load = LegacyVLLMBackend.load_model
    LegacyVLLMBackend.load_model = lambda self, cfg: DummyModel()
    try:
        backend = LegacyVLLMBackend({"model_path": "repo"})
    finally:
        LegacyVLLMBackend.load_model = orig_load
    return backend


class LegacyVLLMBackendConvertOutputTests(unittest.TestCase):
    def test_convert_text(self):
        backend = make_backend()
        out = backend._convert_output({"outputs": [{"text": "hi"}]}, "text")
        self.assertEqual(out, "hi")

    def test_convert_cosyvoice(self):
        backend = make_backend()
        result = {"outputs": [{"audio": [1, 2, 3], "text": "hi"}]}
        out = backend._convert_output(result, "cosyvoice2")
        self.assertEqual(out["type"], "cosyvoice2")
        self.assertEqual(out["raw_outputs"][0]["audio"], [1, 2, 3])

    def test_convert_cosyvoice_custom_obj(self):
        backend = make_backend()

        class Custom:
            def __init__(self):
                self.audio = [9]
                self.note = "ok"

        result = [Custom()]
        out = backend._convert_output(result, "cosyvoice2")
        self.assertEqual(out["raw_outputs"][0]["audio"], [9])
        self.assertEqual(out["raw_outputs"][0]["note"], "ok")


if __name__ == "__main__":
    unittest.main()
