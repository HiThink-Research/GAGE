#!/usr/bin/env python
"""Validate registry_manifest.yaml against current registry + schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from jsonschema import Draft7Validator

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import gage_eval  # noqa: F401
from gage_eval.registry import registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate registry manifest file.")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "registry_manifest.yaml")
    parser.add_argument("--schema", type=Path, default=REPO_ROOT / "docs/registry_manifest.schema.json")
    return parser.parse_args()


def load_schema(path: Path) -> dict:
    return json.loads(path.read_text())


def _normalize(obj):
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    return obj


def main() -> None:
    args = parse_args()
    actual = yaml.safe_load(args.manifest.read_text())
    expected = registry.manifest()
    if _normalize(actual) != _normalize(expected):
        raise SystemExit(
            "registry_manifest.yaml is stale. Run scripts/build_registry_manifest.py and commit the result."
        )
    schema = load_schema(args.schema)
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(actual), key=lambda e: e.path)
    if errors:
        for error in errors:
            print(f"[schema] {list(error.path)}: {error.message}")
        raise SystemExit("registry_manifest.yaml failed schema validation")
    print("registry_manifest.yaml is up to date and valid.")


if __name__ == "__main__":
    main()
