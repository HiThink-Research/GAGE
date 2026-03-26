from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.vllm.runtime_compat import (
    load_vllm_engine_runtime,
    prepare_async_engine_kwargs,
)


class VLLMRuntimeCompatTests(unittest.TestCase):
    def tearDown(self) -> None:
        load_vllm_engine_runtime.cache_clear()

    def test_prepare_async_engine_kwargs_supports_legacy_log_flags(self):
        saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "vllm",
                "vllm.engine",
                "vllm.engine.arg_utils",
                "vllm.engine.async_llm_engine",
            )
        }

        class AsyncEngineArgs:
            def __init__(
                self,
                *,
                model,
                tokenizer=None,
                tensor_parallel_size=1,
                trust_remote_code=True,
                disable_log_requests=False,
                disable_log_stats=False,
                max_model_len=None,
            ):
                self.model = model
                self.tokenizer = tokenizer
                self.tensor_parallel_size = tensor_parallel_size
                self.trust_remote_code = trust_remote_code
                self.disable_log_requests = disable_log_requests
                self.disable_log_stats = disable_log_stats
                self.max_model_len = max_model_len

        class AsyncLLMEngine:
            @classmethod
            def from_engine_args(cls, engine_args):
                return types.SimpleNamespace(engine_args=engine_args)

            async def generate(self, prompt, sampling_params, request_id, **kwargs):
                return prompt, sampling_params, request_id, kwargs

        try:
            sys.modules["vllm"] = types.ModuleType("vllm")
            sys.modules["vllm.engine"] = types.ModuleType("vllm.engine")
            arg_utils = types.ModuleType("vllm.engine.arg_utils")
            arg_utils.AsyncEngineArgs = AsyncEngineArgs
            sys.modules["vllm.engine.arg_utils"] = arg_utils
            async_engine = types.ModuleType("vllm.engine.async_llm_engine")
            async_engine.AsyncLLMEngine = AsyncLLMEngine
            sys.modules["vllm.engine.async_llm_engine"] = async_engine

            runtime = load_vllm_engine_runtime()
            filtered, dropped = prepare_async_engine_kwargs(
                {
                    "model": "repo",
                    "tokenizer": "tok",
                    "tensor_parallel_size": 2,
                    "trust_remote_code": True,
                    "max_model_len": 8192,
                    "unsupported_field": "ignored",
                },
                runtime,
            )

            self.assertEqual(runtime.engine_variant, "legacy")
            self.assertTrue(filtered["disable_log_requests"])
            self.assertTrue(filtered["disable_log_stats"])
            self.assertEqual(filtered["tokenizer"], "tok")
            self.assertEqual(dropped, ("unsupported_field",))
        finally:
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_prepare_async_engine_kwargs_supports_v1_log_flags(self):
        saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "vllm",
                "vllm.engine",
                "vllm.engine.arg_utils",
                "vllm.engine.async_llm_engine",
            )
        }

        class AsyncEngineArgs:
            def __init__(
                self,
                *,
                model,
                tokenizer=None,
                tensor_parallel_size=1,
                trust_remote_code=True,
                enable_log_requests=False,
                disable_log_stats=False,
                limit_mm_per_prompt=None,
            ):
                self.model = model
                self.tokenizer = tokenizer
                self.tensor_parallel_size = tensor_parallel_size
                self.trust_remote_code = trust_remote_code
                self.enable_log_requests = enable_log_requests
                self.disable_log_stats = disable_log_stats
                self.limit_mm_per_prompt = limit_mm_per_prompt

        class AsyncLLMEngine:
            __module__ = "vllm.v1.engine.async_llm"

            @classmethod
            def from_engine_args(cls, engine_args):
                return types.SimpleNamespace(engine_args=engine_args)

            async def generate(self, prompt, sampling_params, request_id, **kwargs):
                return prompt, sampling_params, request_id, kwargs

        try:
            sys.modules["vllm"] = types.ModuleType("vllm")
            sys.modules["vllm.engine"] = types.ModuleType("vllm.engine")
            arg_utils = types.ModuleType("vllm.engine.arg_utils")
            arg_utils.AsyncEngineArgs = AsyncEngineArgs
            sys.modules["vllm.engine.arg_utils"] = arg_utils
            async_engine = types.ModuleType("vllm.engine.async_llm_engine")
            async_engine.AsyncLLMEngine = AsyncLLMEngine
            sys.modules["vllm.engine.async_llm_engine"] = async_engine

            runtime = load_vllm_engine_runtime()
            filtered, dropped = prepare_async_engine_kwargs(
                {
                    "model": "repo",
                    "tokenizer": "tok",
                    "tensor_parallel_size": 1,
                    "trust_remote_code": True,
                    "limit_mm_per_prompt": {"image": 8},
                    "unsupported_field": "ignored",
                },
                runtime,
            )

            self.assertEqual(runtime.engine_variant, "v1")
            self.assertFalse(filtered["enable_log_requests"])
            self.assertTrue(filtered["disable_log_stats"])
            self.assertEqual(filtered["limit_mm_per_prompt"], {"image": 8})
            self.assertEqual(dropped, ("unsupported_field",))
        finally:
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
