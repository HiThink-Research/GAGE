import base64
import io
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
        load_calls = []

        def fake_load(src):
            load_calls.append(list(src))
            return [f"img_{i}_{val}" for i, val in enumerate(src)]

        backend._load_images = fake_load
        payload = {
            "inputs": {"multi_modal_data": {"image": ["path1"], "audio": ["aud1"]}},
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "extra_img"}}]}
            ],
        }
        mm = backend._prepare_multi_modal_data(payload)
        self.assertEqual(load_calls[0], ["extra_img", "path1"])
        self.assertEqual(mm["image"], ["img_0_extra_img", "img_1_path1"])
        self.assertEqual(mm["audio"], ["aud1"])

    def test_prepare_multi_modal_preserves_repeated_message_images(self):
        backend = make_backend()

        def fake_load(src):
            return [f"img_{i}_{val}" for i, val in enumerate(src)]

        backend._load_images = fake_load
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "repeat.png", "detail": "left"}},
                        {"type": "text", "text": "focus on another region"},
                        {"type": "image_url", "image_url": {"url": "repeat.png", "detail": "right"}},
                    ],
                }
            ]
        }
        mm = backend._prepare_multi_modal_data(payload)
        self.assertEqual(mm["image"], ["img_0_repeat.png", "img_1_repeat.png"])

    def test_prepare_multi_modal_dedups_cross_fields(self):
        backend = make_backend()
        load_calls = []

        def fake_load(src):
            load_calls.append(list(src))
            return [f"img_{i}_{val}" for i, val in enumerate(src)]

        backend._load_images = fake_load
        payload = {
            "inputs": {"multi_modal_data": {"image": ["repeat.png", "other.png"]}},
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "repeat.png"}}]}
            ],
        }
        mm = backend._prepare_multi_modal_data(payload)
        # repeat.png should come from messages; other.png is appended from inputs
        self.assertEqual(load_calls[0], ["repeat.png", "other.png"])
        self.assertEqual(mm["image"], ["img_0_repeat.png", "img_1_other.png"])

    def test_prepare_multi_modal_accepts_image_type(self):
        backend = make_backend()

        def fake_load(src):
            return [f"img_{i}_{val}" for i, val in enumerate(src)]

        backend._load_images = fake_load
        payload = {
            "messages": [
                {"role": "user", "content": [{"type": "image", "image": "raw.png"}]},
            ]
        }
        mm = backend._prepare_multi_modal_data(payload)
        self.assertEqual(mm["image"], ["img_0_raw.png"])

    def test_prepare_multi_modal_extracts_audio_from_messages(self):
        backend = make_backend()
        backend._load_images = lambda src: []
        payload = {
            "messages": [
                {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "aud.wav"}}]},
            ]
        }
        mm = backend._prepare_multi_modal_data(payload)
        self.assertEqual(mm["audio"], ["aud.wav"])

    def test_prepare_multi_modal_accepts_input_audio(self):
        backend = make_backend()
        backend._load_images = lambda src: []
        data = base64.b64encode(b"dummy-audio").decode("ascii")
        payload = {
            "messages": [
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}]},
            ]
        }
        mm = backend._prepare_multi_modal_data(payload)
        self.assertEqual(len(mm["audio"]), 1)
        self.assertTrue(isinstance(mm["audio"][0], io.BytesIO))

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
