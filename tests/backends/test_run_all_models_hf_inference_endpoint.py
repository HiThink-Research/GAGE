import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "oneclick" / "backends" / "hf_inference_endpoint" / "run_all_models.sh"


class RunAllModelsScriptTests(unittest.TestCase):
    def test_renders_configs_in_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            matrix_path = tmp / "matrix.yaml"
            gen_dir = tmp / "gen"
            out_dir = tmp / "out"
            matrix_path.write_text(
                """
models:
  - name: unit_demo
    endpoint_name: unit-endpoint
    model_name: repo/model
    reuse_existing: true
    auto_start: false
    max_new_tokens: 8
    max_samples: 1
    concurrency: 1
"""
            )

            env = os.environ.copy()
            env.update(
                {
                    "MODEL_MATRIX": str(matrix_path),
                    "GEN_DIR": str(gen_dir),
                    "OUTPUT_ROOT": str(out_dir),
                    "ENV_FILE": os.devnull,
                    "HUGGINGFACEHUB_API_TOKEN": "dummy",
                    "DRY_RUN": "1",
                }
            )

            result = subprocess.run(
                ["bash", str(SCRIPT)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            rendered = gen_dir / "unit_demo.yaml"
            self.assertTrue(rendered.exists(), "config should be rendered in dry-run mode")
            contents = rendered.read_text()
            self.assertIn("type: hf_inference_endpoint", contents)
            self.assertIn("endpoint_name: unit-endpoint", contents)
            self.assertFalse((out_dir / "unit_demo").exists(), "dry run should skip run.py invocation")
            self.assertIn("DRY_RUN", result.stdout)


if __name__ == "__main__":
    unittest.main()
