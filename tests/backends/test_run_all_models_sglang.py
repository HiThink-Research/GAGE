import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "oneclick" / "backends" / "sglang" / "run_all_models.sh"


class RunAllModelsSGLangTests(unittest.TestCase):
    def test_renders_configs_in_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            matrix_path = tmp / "matrix.yaml"
            gen_dir = tmp / "gen"
            out_dir = tmp / "out"
            matrix_path.write_text(
                """
models:
  - name: unit_sglang
    base_url: http://127.0.0.1:30001
    max_new_tokens: 8
    presence_penalty: 0.1
    max_samples: 2
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

            rendered = gen_dir / "unit_sglang.yaml"
            self.assertTrue(rendered.exists(), "config should be rendered in dry-run mode")
            contents = rendered.read_text()
            self.assertIn("type: sglang", contents)
            self.assertIn("base_url: http://127.0.0.1:30001", contents)
            self.assertIn("presence_penalty: 0.1", contents)
            self.assertFalse(out_dir.exists(), "dry run should skip run.py invocation")
            self.assertIn("DRY_RUN", result.stdout)


if __name__ == "__main__":
    unittest.main()
