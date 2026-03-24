"""On-demand registry imports for dataset-related assets."""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
import re
from typing import Dict

_ASSET_PATTERNS = {
    kind: re.compile(
        rf'registry\.asset\(\s*["\']{kind}["\']\s*,\s*["\']([^"\']+)["\']',
        re.S,
    )
    for kind in (
        "dataset_hubs",
        "dataset_loaders",
        "bundles",
        "dataset_preprocessors",
    )
}


def import_dataset_asset_module(kind: str, asset_name: str) -> None:
    module_name = _dataset_asset_modules(str(kind)).get(str(asset_name))
    if module_name:
        importlib.import_module(module_name)


@lru_cache(maxsize=None)
def _dataset_asset_modules(kind: str) -> Dict[str, str]:
    pattern = _ASSET_PATTERNS.get(str(kind))
    if pattern is None:
        return {}
    root = Path(__file__).resolve().parent
    module_prefix = "gage_eval.assets.datasets"
    mapping: Dict[str, str] = {}
    for path in root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        module_name = _module_name_for_path(path, root=root, module_prefix=module_prefix)
        for asset_name in pattern.findall(source):
            mapping.setdefault(str(asset_name), module_name)
    return mapping


def _module_name_for_path(path: Path, *, root: Path, module_prefix: str) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = relative.parts
    if not parts:
        return module_prefix
    return ".".join((module_prefix, *parts))
