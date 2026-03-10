#!/usr/bin/env python
"""Convert legacy backend_id bindings into inline role adapter configs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate PipelineConfig to inline role adapter backends.")
    parser.add_argument("--input", type=Path, required=True, help="Original YAML config path")
    parser.add_argument("--output", type=Path, help="Output path (defaults to overwrite input)")
    parser.add_argument("--dry-run", action="store_true", help="Print migrated YAML instead of writing it")
    return parser.parse_args()


def migrate(payload: Dict) -> Tuple[Dict, bool]:
    backends = {item.get("backend_id"): item for item in payload.get("backends", []) if isinstance(item, dict)}
    changed = False
    for adapter in payload.get("role_adapters", []):
        backend_id = adapter.get("backend_id")
        if not backend_id or backend_id not in backends:
            continue
        backend_spec = backends[backend_id]
        if "backend" not in adapter:
            adapter["backend"] = {
                "type": backend_spec.get("type"),
                "config": backend_spec.get("config", {}),
            }
        adapter.pop("backend_id", None)
        changed = True
    if changed:
        payload["backends"] = []
    return payload, changed


def main() -> None:
    args = parse_args()
    config_path = args.input
    data = yaml.safe_load(config_path.read_text())
    migrated, changed = migrate(data)
    if not changed:
        print("No backend references detected; nothing to migrate.")
        return
    if args.dry_run:
        print(yaml.safe_dump(migrated, sort_keys=False, allow_unicode=True))
        return
    output = args.output or config_path
    output.write_text(yaml.safe_dump(migrated, sort_keys=False, allow_unicode=True))
    print(f"Migrated config written to {output}")


if __name__ == "__main__":
    main()
