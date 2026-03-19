import base64
import io
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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
        backend = VLLMBackend(config or {"model_path": "repo"})
    finally:
        VLLMBackend.load_model = orig_load
    return backend


class VLLMBackendMultimodalTests(unittest.TestCase):
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
        self.assertEqual(load_calls[0], ["repeat.png", "other.png"])
        self.assertEqual(mm["image"], ["img_0_repeat.png", "img_1_other.png"])

    def test_prepare_multi_modal_accepts_image_type(self):
        backend = make_backend()

        def fake_load(src):
            return [f"img_{i}_{val}" for i, val in enumerate(src)]

        backend._load_images = fake_load
        payload = {"messages": [{"role": "user", "content": [{"type": "image", "image": "raw.png"}]}]}
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

    def test_resolve_mm_limits_prefers_explicit_config(self):
        backend = make_backend({"model_path": "repo", "limit_mm_per_prompt": {"image": 3}})

        with patch.dict("os.environ", {"GAGE_EVAL_VLLM_IMAGE_LIMIT": "7"}, clear=False):
            limits = backend._resolve_mm_limits(backend.config, SimpleNamespace())

        self.assertEqual(limits, {"image": 3})

    def test_resolve_mm_limits_uses_env_fallback(self):
        backend = make_backend()

        with patch.dict(
            "os.environ",
            {"GAGE_EVAL_VLLM_IMAGE_LIMIT": "4", "GAGE_EVAL_VLLM_AUDIO_LIMIT": "2"},
            clear=False,
        ):
            limits = backend._resolve_mm_limits(backend.config, SimpleNamespace())

        self.assertEqual(limits, {"image": 4, "audio": 2})

    def test_resolve_mm_limits_returns_none_without_explicit_boundary(self):
        backend = make_backend()

        with patch.dict("os.environ", {}, clear=True):
            limits = backend._resolve_mm_limits(backend.config, SimpleNamespace())

        self.assertIsNone(limits)

    def test_build_engine_forwards_explicit_mm_limits(self):
        backend = make_backend({"model_path": "repo", "limit_mm_per_prompt": {"image": 8}})
        args = SimpleNamespace(
            tokenizer=None,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enforce_eager=None,
            pipeline_parallel_size=None,
            data_parallel_size=None,
            data_parallel_rank=None,
            data_parallel_size_local=None,
            data_parallel_address=None,
            data_parallel_rpc_port=None,
            data_parallel_backend=None,
            distributed_executor_backend=None,
            enable_expert_parallel=None,
            max_length=None,
            block_size=None,
            num_gpu_blocks=None,
            num_cpu_blocks=None,
            forced_num_gpu_blocks=None,
            num_gpu_blocks_override=None,
            limit_mm_per_prompt={"image": 8},
        )

        class FakeAsyncEngineArgs:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class FakeAsyncLLMEngine:
            __module__ = "vllm.engine.async_llm_engine"

            @classmethod
            def from_engine_args(cls, engine_args):
                return types.SimpleNamespace(engine_args=engine_args, engine=types.SimpleNamespace())

        runtime = SimpleNamespace(
            engine_variant="test",
            async_engine_args_cls=FakeAsyncEngineArgs,
            async_llm_engine_cls=FakeAsyncLLMEngine,
        )

        with (
            patch("gage_eval.role.model.backends.vllm_backend.load_vllm_engine_runtime", return_value=runtime),
            patch(
                "gage_eval.role.model.backends.vllm_backend.prepare_async_engine_kwargs",
                side_effect=lambda kwargs, _: (kwargs, ()),
            ),
            patch("gage_eval.role.model.backends.vllm_backend._prime_vllm_engine_renderer_state", return_value=None),
        ):
            engine, _ = backend._build_engine(
                types.SimpleNamespace(model_type="vision"),
                processor=None,
                model_id="repo",
                args=args,
                config=backend.config,
            )

        self.assertEqual(engine.engine_args.limit_mm_per_prompt, {"image": 8})


if __name__ == "__main__":
    unittest.main()
