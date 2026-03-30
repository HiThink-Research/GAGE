#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "gage_eval" / "tools" / "loguru_style_checker.py"
SPEC = importlib.util.spec_from_file_location("gage_eval_loguru_style_checker", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - defensive guard
    raise SystemExit(f"Failed to load loguru style checker from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
main = MODULE.main


def _resolve_cli_paths(raw_args: list[str]) -> list[str]:
    if not raw_args:
        return [str(ROOT / "src" / "gage_eval")]
    resolved: list[str] = []
    for item in raw_args:
        path = Path(item)
        resolved.append(str(path if path.is_absolute() else (ROOT / path).resolve()))
    return resolved


if __name__ == "__main__":
    raise SystemExit(main(_resolve_cli_paths(sys.argv[1:])))
