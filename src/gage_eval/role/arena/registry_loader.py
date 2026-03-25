"""On-demand registry imports for arena assets."""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
import re
from typing import Dict, Iterable

_ASSET_PATTERNS = {
    kind: re.compile(
        rf'registry\.asset\(\s*["\']{kind}["\']\s*,\s*["\']([^"\']+)["\']',
        re.S,
    )
    for kind in ("arena_impls", "arena_game_providers", "parser_impls", "renderer_impls")
}
_SKIP_DIR_NAMES = {"__pycache__", "tests"}
_ASSET_SEARCH_ROOTS = {
    "arena_impls": ("games",),
    "arena_game_providers": ("game_providers.py",),
    "parser_impls": ("parsers", "games"),
    "renderer_impls": ("games",),
}


def import_arena_asset_module(kind: str, asset_name: str) -> None:
    module_name = _arena_asset_modules(str(kind)).get(str(asset_name))
    if module_name:
        importlib.import_module(module_name)


def import_all_arena_asset_modules(kind: str) -> None:
    for module_name in dict.fromkeys(_arena_asset_modules(str(kind)).values()):
        importlib.import_module(module_name)


@lru_cache(maxsize=None)
def _arena_asset_modules(kind: str) -> Dict[str, str]:
    pattern = _ASSET_PATTERNS.get(str(kind))
    if pattern is None:
        return {}
    root = Path(__file__).resolve().parent
    module_prefix = "gage_eval.role.arena"
    mapping: Dict[str, str] = {}
    for path in _iter_asset_source_paths(root=root, kind=str(kind)):
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


def _asset_search_roots(*, root: Path, kind: str) -> tuple[Path, ...]:
    relative_roots = _ASSET_SEARCH_ROOTS.get(str(kind), ())
    return tuple((root / relative_root).resolve() for relative_root in relative_roots)


def _iter_asset_source_paths(*, root: Path, kind: str) -> Iterable[Path]:
    for search_root in _asset_search_roots(root=root, kind=kind):
        if search_root.is_file():
            yield search_root
            continue
        if not search_root.is_dir():
            continue
        for path in search_root.rglob("*.py"):
            if _SKIP_DIR_NAMES.intersection(path.parts):
                continue
            yield path


def _module_name_for_path(path: Path, *, root: Path, module_prefix: str) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = relative.parts
    if not parts:
        return module_prefix
    return ".".join((module_prefix, *parts))
