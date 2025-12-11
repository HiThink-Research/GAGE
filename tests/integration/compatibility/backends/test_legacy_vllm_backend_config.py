from pathlib import Path
import yaml
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class LegacyVLLMBackendConfigTests(unittest.TestCase):
    def test_text_config_backend_type(self):
        cfg = load_yaml(PROJECT_ROOT / "config" / "custom" / "piqa_legacy_vllm_async.yaml")
        backend = cfg["backends"][0]
        self.assertEqual(backend["type"], "legacy_vllm")
        self.assertIn("model_path", backend["config"])

    def test_multimodal_config_backend_type(self):
        cfg = load_yaml(PROJECT_ROOT / "config" / "custom" / "mmmu_legacy_vllm.yaml")
        backend = cfg["backends"][0]
        self.assertEqual(backend["type"], "legacy_vllm")
        # dataset should include doc_to_visual
        dataset = cfg["datasets"][0]
        self.assertIn("doc_to_visual", dataset["params"])

    def test_run_configs_reference_custom(self):
        piqa_run = load_yaml(PROJECT_ROOT / "config" / "run_configs" / "piqa_legacy_vllm_async_run_1.yaml")
        self.assertEqual(piqa_run["base_task"], "custom/piqa_legacy_vllm_async")
        mmmu_run = load_yaml(PROJECT_ROOT / "config" / "run_configs" / "mmmu_legacy_vllm_run_1.yaml")
        self.assertEqual(mmmu_run["base_task"], "custom/mmmu_legacy_vllm")


if __name__ == "__main__":
    unittest.main()
