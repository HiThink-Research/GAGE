import yaml
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ConfigFieldMappingTests(unittest.TestCase):
    def _load(self, relative):
        return yaml.safe_load((ROOT / relative).read_text())

    def test_docvqa_preprocess_kwargs(self):
        cfg = self._load("config/custom/docvqa_qwen_vl.yaml")
        kw = cfg["datasets"][0]["params"]["preprocess_kwargs"]
        self.assertIn("content_field", kw)
        self.assertNotIn("content_root", kw)
        doc_kwargs = cfg["datasets"][0]["params"].get("doc_to_visual_kwargs")
        self.assertIsNone(doc_kwargs)

    def test_mmmu_doc_to_visual_kwargs(self):
        paths = [
            "config/custom/mmmu_qwen_vl.yaml",
            "config/custom/mmmu_local_vlm.yaml",
            "config/custom/mmmu_legacy_vllm.yaml",
        ]
        for rel in paths:
            cfg = self._load(rel)
            kw = cfg["datasets"][0]["params"]["doc_to_visual_kwargs"]
            self.assertIn("content_field", kw)
            self.assertTrue("content_root" not in kw or kw.get("content_root"))


if __name__ == "__main__":
    unittest.main()
