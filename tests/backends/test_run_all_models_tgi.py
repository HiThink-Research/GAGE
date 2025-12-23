import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "oneclick" / "backends" / "tgi" / "run_all_models.sh"


class RunAllModelsTGITests(unittest.TestCase):
    def test_renders_configs_in_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            matrix_path = tmp / "matrix.yaml"
            gen_dir = tmp / "gen"
            out_dir = tmp / "out"
            matrix_path.write_text(
                """
models:
  - name: unit_tgi
    base_url: http://127.0.0.1:8081
    max_new_tokens: 8
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

            rendered = gen_dir / "unit_tgi.yaml"
            self.assertTrue(rendered.exists(), "config should be rendered in dry-run mode")
            contents = rendered.read_text()
            self.assertIn("type: tgi", contents)
            self.assertIn("base_url: http://127.0.0.1:8081", contents)
            self.assertFalse(out_dir.exists(), "dry run should skip run.py invocation")
            self.assertIn("DRY_RUN", result.stdout)


if __name__ == "__main__":
    unittest.main()
