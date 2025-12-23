#!/usr/bin/env python
"""Generate registry_manifest.yaml from the discovered assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import gage_eval  # noqa: F401 - triggers auto-discovery
from gage_eval.registry import registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build registry manifest for gage-eval.")
    parser.add_argument("--out", type=Path, default=Path("registry_manifest.yaml"), help="Output manifest path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = registry.manifest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(manifest, sort_keys=True, allow_unicode=True), encoding="utf-8")
    print(f"Wrote registry manifest to {args.out}")


if __name__ == "__main__":
    main()
