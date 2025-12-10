import sys
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import gage_eval.role.model.backends.legacy_vllm_backend as legacy_mod
from gage_eval.role.model.backends.legacy_vllm_backend import LegacyVLLMBackend


class DummyModel:
    pass


def make_backend(config=None):
    orig_load = LegacyVLLMBackend.load_model
    LegacyVLLMBackend.load_model = lambda self, cfg: DummyModel()
    try:
        backend = LegacyVLLMBackend(config or {"model_path": "repo"})
    finally:
        LegacyVLLMBackend.load_model = orig_load
    return backend


class LegacyVLLMBackendMultimodalTests(unittest.TestCase):
    def test_prepare_multi_modal_combines_sources(self):
        backend = make_backend()
        # patch _load_images to avoid real IO
        backend._load_images = lambda src: ["img_stub"] if src else []
        payload = {
            "inputs": {"multi_modal_data": {"image": ["path1"], "audio": ["aud1"]}},
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "extra_img"}}]}
            ],
        }
        mm = backend._prepare_multi_modal_data(payload)
        self.assertEqual(mm["image"], ["img_stub", "img_stub"])
        self.assertEqual(mm["audio"], ["aud1"])

    def test_load_multimodal_payload_uses_loader(self):
        calls = {}

        def fake_loader(processor, image=None, audio=None, return_dict=False, resize_opts=None):
            calls["called"] = True
            return {"image": image, "audio": audio}

        backend = make_backend()
        backend._processor = "proc"
        legacy_mod.load_multimodal_data, orig_loader = fake_loader, legacy_mod.load_multimodal_data
        try:
            mm = backend._load_multimodal_payload({"image": ["img"], "audio": ["aud"]})
        finally:
            legacy_mod.load_multimodal_data = orig_loader
        self.assertEqual(calls.get("called"), True)
        self.assertEqual(mm, {"image": ["img"], "audio": ["aud"]})


if __name__ == "__main__":
    unittest.main()
