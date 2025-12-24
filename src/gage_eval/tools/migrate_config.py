"""CLI tool that migrates legacy PipelineConfig YAMLs to the new backends/backend_id layout.

This tool focuses on structural migration: extract inline backends from RoleAdapters into the top-level `backends`
list and replace inline configs with `backend_id` references. It intentionally does not modify numeric fields
(e.g., concurrency, max_batch_size) to avoid introducing implicit behavior changes.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config '{path}' must be a mapping at the top level")
    return data


def _dump_yaml(payload: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _ensure_list(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return list(obj)
    raise ValueError("expected list when normalizing config payload")


def migrate_config_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new payload where inline backends are extracted to top-level backends.

    Migration rules:
    - Collect all `role_adapters[].backend` entries and build a top-level `backends` list.
    - Generate a stable `backend_id` per unique backend (reuse `adapter_id` when possible).
    - Replace `role_adapters[].backend` with a `backend_id` reference.
    - Preserve field order and numeric config values; do not change semantics.
    """

    data = deepcopy(payload)
    role_adapters = _ensure_list(data.get("role_adapters"))
    existing_backends = _ensure_list(data.get("backends"))

    # Build a stable dedup key: backend type + normalized JSON config.
    backend_index: Dict[str, str] = {}
    backend_specs: Dict[str, Dict[str, Any]] = {}

    for entry in existing_backends:
        backend_id = entry.get("backend_id")
        if not backend_id:
            continue
        backend_type = entry.get("type")
        config = entry.get("config") or {}
        key = f"{backend_type}:{json.dumps(config, sort_keys=True, separators=(',', ':'), ensure_ascii=False)}"
        backend_index[key] = backend_id
        backend_specs[backend_id] = entry

    for adapter in role_adapters:
        inline_backend = adapter.get("backend")
        backend_id = adapter.get("backend_id")
        if not inline_backend or backend_id:
            # Skip adapters that already use backend_id or do not have an inline backend.
            continue
        if not isinstance(inline_backend, dict):
            raise ValueError(
                f"role_adapter '{adapter.get('adapter_id')}' inline backend must be a mapping during migration"
            )
        backend_type = inline_backend.get("type")
        if not backend_type:
            raise ValueError(
                f"role_adapter '{adapter.get('adapter_id')}' inline backend missing 'type' during migration"
            )
        config = inline_backend.get("config") or {}
        key = f"{backend_type}:{json.dumps(config, sort_keys=True, separators=(',', ':'), ensure_ascii=False)}"
        if key in backend_index:
            new_backend_id = backend_index[key]
        else:
            # Prefer using adapter_id as backend_id for readability; append a suffix if already taken.
            base_id = adapter.get("adapter_id") or backend_type
            new_backend_id = base_id
            suffix = 1
            existing_ids = {spec.get("backend_id") for spec in existing_backends}
            while new_backend_id in existing_ids:
                suffix += 1
                new_backend_id = f"{base_id}_{suffix}"
            backend_index[key] = new_backend_id
            backend_specs[new_backend_id] = {
                "backend_id": new_backend_id,
                "type": backend_type,
                "config": inline_backend.get("config", {}) or {},
            }
            existing_backends.append(backend_specs[new_backend_id])
        # Replace the inline backend with a backend_id reference on the adapter.
        adapter["backend_id"] = new_backend_id
        adapter.pop("backend", None)

    if existing_backends:
        data["backends"] = existing_backends
    if role_adapters:
        data["role_adapters"] = role_adapters
    return data


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate legacy gage-eval configs to use top-level backends + backend_id references.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the legacy YAML config.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the migrated YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print a short migration summary without writing the output file.",
    )
    return parser.parse_args(list(argv) if argv is not None else [])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or [])
    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()

    payload = _load_yaml(src)
    migrated = migrate_config_payload(payload)

    if args.dry_run:
        print(f"[gage-eval] dry-run migration summary for {src}:")
        orig_adapters = _ensure_list(payload.get("role_adapters"))
        new_adapters = _ensure_list(migrated.get("role_adapters"))
        orig_backends = _ensure_list(payload.get("backends"))
        new_backends = _ensure_list(migrated.get("backends"))
        print(f"  - role_adapters: {len(orig_adapters)} -> {len(new_adapters)}")
        print(f"  - backends: {len(orig_backends)} -> {len(new_backends)}")
        return 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    _dump_yaml(migrated, dst)
    print(f"[gage-eval] migrated config written to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
