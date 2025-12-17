"""CLI tool that migrates legacy PipelineConfig YAMLs to the new backends/backend_id layout.

该工具聚焦「结构迁移」：将 RoleAdapter 内联 backend 提取到顶层 backends 列表，并用 backend_id
替换原有 inline 配置。不会修改任何数值型字段（如 concurrency、max_batch_size），以避免引入隐式行为变化。
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

    迁移规则：
    - 收集所有 `role_adapters[].backend`，构造顶层 backends 列表；
    - 为每个唯一 backend 生成稳定的 `backend_id`（若已有则复用 adapter_id）；
    - 将 `role_adapters[].backend` 替换为 `backend_id` 引用；
    - 保留原有字段顺序与数值型配置，不做语义调整。
    """

    data = deepcopy(payload)
    role_adapters = _ensure_list(data.get("role_adapters"))
    existing_backends = _ensure_list(data.get("backends"))

    # 建立一个稳定的去重 key：基于 backend type + config 的规范化 JSON 表示。
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
            # 已经使用 backend_id 或没有 inline backend，直接跳过。
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
            # 优先复用 adapter_id 作为 backend_id，便于人类阅读；如已被占用则追加后缀。
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
        # 替换 adapter 上的 backend 为 backend_id 引用。
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
