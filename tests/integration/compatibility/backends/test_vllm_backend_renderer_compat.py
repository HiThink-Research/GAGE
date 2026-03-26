import importlib.util
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4] / "src"
COMPAT_PATH = ROOT / "gage_eval" / "compat" / "vllm_renderer_patch.py"
COMPAT_SPEC = importlib.util.spec_from_file_location("gage_eval_vllm_renderer_patch_test", COMPAT_PATH)
if COMPAT_SPEC is None or COMPAT_SPEC.loader is None:
    raise RuntimeError(f"Failed to load compat module from {COMPAT_PATH}")
compat_mod = importlib.util.module_from_spec(COMPAT_SPEC)
COMPAT_SPEC.loader.exec_module(compat_mod)


class VLLMRendererCompatTests(unittest.TestCase):
    def test_install_patch_backfills_missing_mm_state(self):
        saved_modules = {name: sys.modules.get(name) for name in (
            "vllm",
            "vllm.renderers",
            "vllm.renderers.base",
            "vllm.utils",
            "vllm.utils.counter",
            "vllm.multimodal",
            "vllm.multimodal.registry",
        )}

        class DummyCounter:
            def __init__(self):
                self.value = 0

            def inc(self, step):
                self.value += step
                return self.value

        class DummyTimingRegistry:
            def __init__(self, observability_config):
                self.observability_config = observability_config

            def get(self, request_id):
                return f"timing:{request_id}:{self.observability_config}"

        class BaseRenderer:
            def __init__(self):
                self.config = types.SimpleNamespace(observability_config="obs")

            def _process_multimodal(self):
                mm_req_id = f"renderer-mm-{self._mm_req_counter.inc(1)}"
                return mm_req_id, self._mm_timing_registry.get(mm_req_id)

        try:
            sys.modules["vllm"] = types.ModuleType("vllm")
            sys.modules["vllm.renderers"] = types.ModuleType("vllm.renderers")
            renderers_base = types.ModuleType("vllm.renderers.base")
            renderers_base.BaseRenderer = BaseRenderer
            sys.modules["vllm.renderers.base"] = renderers_base

            sys.modules["vllm.utils"] = types.ModuleType("vllm.utils")
            utils_counter = types.ModuleType("vllm.utils.counter")
            utils_counter.AtomicCounter = DummyCounter
            sys.modules["vllm.utils.counter"] = utils_counter

            sys.modules["vllm.multimodal"] = types.ModuleType("vllm.multimodal")
            mm_registry = types.ModuleType("vllm.multimodal.registry")
            mm_registry.MultiModalTimingRegistry = DummyTimingRegistry
            sys.modules["vllm.multimodal.registry"] = mm_registry

            compat_mod.install_vllm_renderer_compat_patches()

            renderer = BaseRenderer()
            request_id, timing = renderer._process_multimodal()

            self.assertEqual(request_id, "renderer-mm-1")
            self.assertEqual(timing, "timing:renderer-mm-1:obs")
            self.assertTrue(hasattr(renderer, "_mm_req_counter"))
            self.assertTrue(hasattr(renderer, "_mm_timing_registry"))
        finally:
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_install_patch_updates_hf_renderer_alias_and_mm_processor(self):
        saved_modules = {name: sys.modules.get(name) for name in (
            "vllm",
            "vllm.renderers",
            "vllm.renderers.base",
            "vllm.renderers.hf",
            "vllm.renderers.hf_renderer",
            "vllm.utils",
            "vllm.utils.counter",
            "vllm.utils.torch_utils",
            "vllm.multimodal",
            "vllm.multimodal.registry",
            "vllm.v1",
            "vllm.v1.metrics",
            "vllm.v1.metrics.stats",
        )}

        class DummyCounter:
            def __init__(self):
                self.value = 0

            def inc(self, step):
                self.value += step
                return self.value

        class DummyTimingRegistry:
            def __init__(self, observability_config):
                self.observability_config = observability_config

            def get(self, request_id):
                return f"timing:{request_id}:{self.observability_config}"

        class DummyCacheStats:
            pass

        class DummyMMRegistry:
            def processor_cache_from_config(self, _config):
                return "cache"

            def create_processor(self, _model_config, *, tokenizer, cache):
                return types.SimpleNamespace(name=f"processor:{cache}", tokenizer=tokenizer)

        class BaseRenderer:
            def __init__(self):
                self.config = types.SimpleNamespace(
                    observability_config="obs",
                    model_config=types.SimpleNamespace(model_type="qwen2_vl"),
                    parallel_config=types.SimpleNamespace(api_process_rank=7),
                )
                self.model_config = self.config.model_config
                self.tokenizer = types.SimpleNamespace(name="tok")

            def get_mm_processor(self):
                processor = getattr(self, "mm_processor", None)
                if processor is None:
                    raise RuntimeError("missing mm_processor")
                return processor

            def _process_multimodal(self):
                mm_req_id = f"renderer-mm-{self.api_process_rank}-{self._mm_req_counter.inc(1)}"
                timing = self._mm_timing_registry.get(mm_req_id)
                return mm_req_id, self.get_mm_processor().name, timing

        class HfRenderer(BaseRenderer):
            _process_multimodal = BaseRenderer._process_multimodal

        try:
            sys.modules["vllm"] = types.ModuleType("vllm")
            sys.modules["vllm.renderers"] = types.ModuleType("vllm.renderers")

            renderers_base = types.ModuleType("vllm.renderers.base")
            renderers_base.BaseRenderer = BaseRenderer
            sys.modules["vllm.renderers.base"] = renderers_base

            renderers_hf = types.ModuleType("vllm.renderers.hf")
            renderers_hf.HfRenderer = HfRenderer
            sys.modules["vllm.renderers.hf"] = renderers_hf

            renderers_hf_legacy = types.ModuleType("vllm.renderers.hf_renderer")
            renderers_hf_legacy.HfRenderer = HfRenderer
            sys.modules["vllm.renderers.hf_renderer"] = renderers_hf_legacy

            sys.modules["vllm.utils"] = types.ModuleType("vllm.utils")

            utils_counter = types.ModuleType("vllm.utils.counter")
            utils_counter.AtomicCounter = DummyCounter
            sys.modules["vllm.utils.counter"] = utils_counter

            utils_torch = types.ModuleType("vllm.utils.torch_utils")
            utils_torch.set_default_torch_num_threads = nullcontext
            sys.modules["vllm.utils.torch_utils"] = utils_torch

            multimodal_mod = types.ModuleType("vllm.multimodal")
            multimodal_mod.MULTIMODAL_REGISTRY = DummyMMRegistry()
            sys.modules["vllm.multimodal"] = multimodal_mod

            mm_registry = types.ModuleType("vllm.multimodal.registry")
            mm_registry.MultiModalTimingRegistry = DummyTimingRegistry
            sys.modules["vllm.multimodal.registry"] = mm_registry

            sys.modules["vllm.v1"] = types.ModuleType("vllm.v1")
            sys.modules["vllm.v1.metrics"] = types.ModuleType("vllm.v1.metrics")
            v1_stats = types.ModuleType("vllm.v1.metrics.stats")
            v1_stats.MultiModalCacheStats = DummyCacheStats
            sys.modules["vllm.v1.metrics.stats"] = v1_stats

            compat_mod.install_vllm_renderer_compat_patches()

            renderer = HfRenderer()
            request_id, processor_name, timing = renderer._process_multimodal()

            self.assertEqual(request_id, "renderer-mm-7-1")
            self.assertEqual(processor_name, "processor:cache")
            self.assertEqual(timing, "timing:renderer-mm-7-1:obs")
            self.assertEqual(renderer.api_process_rank, 7)
            self.assertIsInstance(renderer._mm_cache_stats, DummyCacheStats)
            self.assertTrue(compat_mod.detect_vllm_engine_multimodal_support(renderer))

            primed_renderer = HfRenderer()
            engine = types.SimpleNamespace(
                input_processor=types.SimpleNamespace(
                    input_preprocessor=types.SimpleNamespace(renderer=primed_renderer),
                )
            )
            compat_mod.prime_vllm_engine_renderer_state(engine, require_mm_processor=True)

            self.assertTrue(hasattr(primed_renderer, "_mm_req_counter"))
            self.assertEqual(primed_renderer.mm_processor.name, "processor:cache")
        finally:
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_detect_multimodal_support_returns_false_for_text_only_renderer(self):
        saved_modules = {name: sys.modules.get(name) for name in (
            "vllm",
            "vllm.renderers",
            "vllm.renderers.base",
            "vllm.utils",
            "vllm.utils.counter",
            "vllm.multimodal",
            "vllm.multimodal.registry",
        )}

        class DummyCounter:
            def __init__(self):
                self.value = 0

            def inc(self, step):
                self.value += step
                return self.value

        class DummyTimingRegistry:
            def __init__(self, observability_config):
                self.observability_config = observability_config

            def get(self, request_id):
                return f"timing:{request_id}:{self.observability_config}"

        class TextOnlyRenderer:
            def __init__(self):
                self.config = types.SimpleNamespace(observability_config="obs")

            def get_mm_processor(self):
                raise ValueError("Multi-modal processor not available for text-only models")

            def _process_multimodal(self):
                raise AssertionError("not called in this test")

        try:
            sys.modules["vllm"] = types.ModuleType("vllm")
            sys.modules["vllm.renderers"] = types.ModuleType("vllm.renderers")
            renderers_base = types.ModuleType("vllm.renderers.base")
            renderers_base.BaseRenderer = TextOnlyRenderer
            sys.modules["vllm.renderers.base"] = renderers_base

            sys.modules["vllm.utils"] = types.ModuleType("vllm.utils")
            utils_counter = types.ModuleType("vllm.utils.counter")
            utils_counter.AtomicCounter = DummyCounter
            sys.modules["vllm.utils.counter"] = utils_counter

            sys.modules["vllm.multimodal"] = types.ModuleType("vllm.multimodal")
            mm_registry = types.ModuleType("vllm.multimodal.registry")
            mm_registry.MultiModalTimingRegistry = DummyTimingRegistry
            sys.modules["vllm.multimodal.registry"] = mm_registry

            compat_mod.install_vllm_renderer_compat_patches()

            engine = types.SimpleNamespace(renderer=TextOnlyRenderer())
            compat_mod.prime_vllm_engine_renderer_state(engine, require_mm_processor=False)

            self.assertFalse(compat_mod.detect_vllm_engine_multimodal_support(engine))
        finally:
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
