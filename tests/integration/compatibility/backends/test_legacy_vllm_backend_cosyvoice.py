import sys
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.legacy_vllm_backend import LegacyVLLMBackend


class DummyCosyVoice2Model:
    def __init__(self, model_dir, flow, hift, fp16):
        self.model_dir = model_dir
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.loaded = []

    def load(self, flow_path, hift_path):
        self.loaded.append(("load", flow_path, hift_path))

    def load_jit(self, path):
        self.loaded.append(("load_jit", path))

    def load_trt(self, plan_path, onnx_path, fp16):
        self.loaded.append(("load_trt", plan_path, onnx_path, fp16))

    def generate(self, **kwargs):
        return {"outputs": [{"text": "cosyvoice2"}]}


def write_yaml(path: Path):
    content = "flow: flow_cfg\nhift: hift_cfg\nget_tokenizer: !!python/name:builtins.str"
    path.write_text(content, encoding="utf-8")


class LegacyVLLMBackendCosyVoiceTests(unittest.TestCase):
    def test_cosyvoice_branch_loads_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "cosyvoice2.yaml"
            write_yaml(yaml_path)

            dummy_module = types.SimpleNamespace(CosyVoice2Model=DummyCosyVoice2Model)

            def fake_load_hyperpyyaml(f, overrides=None):
                return {"flow": "flow_cfg", "hift": "hift_cfg", "get_tokenizer": lambda: "tok"}

            sys.modules["utils.cosyvoice.model"] = dummy_module
            sys.modules["hyperpyyaml"] = types.SimpleNamespace(load_hyperpyyaml=fake_load_hyperpyyaml)
            try:
                cfg = {"model_path": tmpdir, "output_type": "cosyvoice2", "fp16": True, "load_trt": True, "load_jit": True}
                backend = LegacyVLLMBackend(cfg)
            finally:
                sys.modules.pop("utils.cosyvoice.model", None)
                sys.modules.pop("hyperpyyaml", None)

            self.assertIsInstance(backend.model, DummyCosyVoice2Model)
            self.assertEqual(backend._processor, "tok")
            self.assertTrue(any(call[0] == "load" for call in backend.model.loaded))
            self.assertTrue(any(call[0] == "load_jit" for call in backend.model.loaded))
            self.assertTrue(any(call[0] == "load_trt" for call in backend.model.loaded))


if __name__ == "__main__":
    unittest.main()
