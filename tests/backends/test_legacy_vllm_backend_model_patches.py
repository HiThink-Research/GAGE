import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.legacy_vllm_backend import LegacyVLLMBackend


class DummyConfig:
    def __init__(self):
        self.architectures = ["RewardModel"]
        self.model_type = "llama"
        self.rope_scaling = {}
        self.max_position_embeddings = 4096
        self.seq_length = 0
        self.text_config = SimpleNamespace(
            max_position_embeddings=4096,
            model_type="llama",
            rope_scaling={},
        )
        self.moe_intermediate_size = 1


class DummyProcessor:
    pass


def make_backend_with_patches(config_overrides=None):
    cfg = {"model_path": "repo", "max_model_len": 8192, "low_vram": True, "block_size": 32, "tensor_parallel_size": 2}
    if config_overrides:
        cfg.update(config_overrides)

    # Patch loader to return dummy config/processor and dummy engine
    orig_load_cfg = LegacyVLLMBackend._load_auto_config
    orig_load_proc = LegacyVLLMBackend._load_auto_processor
    orig_build_engine = LegacyVLLMBackend._build_engine

    LegacyVLLMBackend._load_auto_config = lambda self, mid, trust_remote_code: DummyConfig()
    LegacyVLLMBackend._load_auto_processor = lambda self, mid, trust_remote_code: DummyProcessor()

    def fake_build_engine(self, cfg_obj, processor, model_id, args_ns):
        class DummyEngine:
            def __init__(self, cfg_obj, args_ns, processor):
                self.cfg = cfg_obj
                self.args = args_ns
                self.processor = processor

            def generate(self, **kwargs):
                return {"outputs": [{"text": "ok"}]}

            def abort(self, *_args, **_kwargs):
                return None

        return DummyEngine(cfg_obj, args_ns, processor), processor

    LegacyVLLMBackend._build_engine = fake_build_engine

    try:
        backend = LegacyVLLMBackend(cfg)
    finally:
        LegacyVLLMBackend._load_auto_config = orig_load_cfg
        LegacyVLLMBackend._load_auto_processor = orig_load_proc
        LegacyVLLMBackend._build_engine = orig_build_engine
    return backend


class LegacyVLLMBackendModelPatchesTests(unittest.TestCase):
    def test_reward_moe_rope_low_vram_patched(self):
        backend = make_backend_with_patches()
        engine = backend.model
        cfg = engine.cfg
        args = engine.args

        # Reward patch
        self.assertIn("LlamaForRewardModel", cfg.architectures)
        # rope scaling update to max_model_len
        self.assertEqual(cfg.text_config.max_position_embeddings, 8192)
        # low vram cache blocks applied
        self.assertIsNotNone(getattr(args, "num_gpu_blocks"))
        self.assertGreater(getattr(args, "num_gpu_blocks"), 0)
        # MoE expert parallel toggled
        self.assertTrue(getattr(args, "enable_expert_parallel"))


if __name__ == "__main__":
    unittest.main()
