import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.tools.config_checker import validate_config


def test_gomoku_template_valid():
    template_path = ROOT / "config" / "builtin_templates" / "gomoku_local_demo" / "v1.yaml"
    validate_config(template_path)
