#!/usr/bin/env python3
"""Generate registry discovery manifests from static source analysis."""

from __future__ import annotations

import argparse
import ast
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "gage_eval"
MANIFEST_ROOT = PACKAGE_ROOT / "registry" / "manifests"
MANIFEST_VERSION = 1

SHARD_BY_KIND = {
    "arena_impls": "arena",
    "arena_game_providers": "arena",
    "parser_impls": "arena",
    "renderer_impls": "arena",
    "backends": "backends",
    "dataset_hubs": "datasets",
    "dataset_loaders": "datasets",
    "bundles": "datasets",
    "dataset_preprocessors": "datasets",
    "metrics": "metrics",
}
DEFAULT_SHARD = "core"
LOAD_PHASE_BY_KIND = {
    "pipeline_steps": "baseline",
    "summary_generators": "baseline",
    "prompts": "baseline",
    "reporting_sinks": "baseline",
    "observability_plugins": "baseline",
}

MANUAL_OVERRIDES = (
    {
        "kind": "prompts",
        "name": "dut/general@v1",
        "module": "gage_eval.assets.prompts.catalog",
        "load_phase": "baseline",
        "declared_in": "src/gage_eval/assets/prompts/catalog/__init__.py",
        "aliases": [],
        "optional": False,
    },
    {
        "kind": "prompts",
        "name": "judge/general@v1",
        "module": "gage_eval.assets.prompts.catalog",
        "load_phase": "baseline",
        "declared_in": "src/gage_eval/assets/prompts/catalog/__init__.py",
        "aliases": [],
        "optional": False,
    },
    {
        "kind": "prompts",
        "name": "dut/appworld@v1",
        "module": "gage_eval.assets.prompts.catalog",
        "load_phase": "baseline",
        "declared_in": "src/gage_eval/assets/prompts/catalog/__init__.py",
        "aliases": [],
        "optional": False,
    },
    {
        "kind": "prompts",
        "name": "helper/appworld_api_predictor@v1",
        "module": "gage_eval.assets.prompts.catalog",
        "load_phase": "baseline",
        "declared_in": "src/gage_eval/assets/prompts/catalog/__init__.py",
        "aliases": [],
        "optional": False,
    },
)


@dataclass(frozen=True, slots=True)
class AssetRecord:
    kind: str
    name: str
    module: str
    declared_in: str
    load_phase: str

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "module": self.module,
            "load_phase": self.load_phase,
            "declared_in": self.declared_in,
            "aliases": [],
            "optional": False,
        }


class AssetCollector(ast.NodeVisitor):
    def __init__(self, *, path: Path, module_name: str) -> None:
        self._path = path
        self._module_name = module_name
        self.records: list[AssetRecord] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._collect_decorators(node.decorator_list)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._collect_decorators(node.decorator_list)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._collect_decorators(node.decorator_list)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        parsed = _parse_asset_call(node)
        if parsed is not None:
            kind, name = parsed
            self.records.append(
                AssetRecord(
                    kind=kind,
                    name=name,
                    module=self._module_name,
                    declared_in=_declared_in(self._path),
                    load_phase=LOAD_PHASE_BY_KIND.get(kind, "prepare_only"),
                )
            )
        self.generic_visit(node)

    def _collect_decorators(self, decorators: Iterable[ast.expr]) -> None:
        for decorator in decorators:
            if not isinstance(decorator, ast.Call):
                continue
            parsed = _parse_asset_call(decorator)
            if parsed is None:
                continue
            kind, name = parsed
            self.records.append(
                AssetRecord(
                    kind=kind,
                    name=name,
                    module=self._module_name,
                    declared_in=_declared_in(self._path),
                    load_phase=LOAD_PHASE_BY_KIND.get(kind, "prepare_only"),
                )
            )


def _parse_asset_call(node: ast.Call) -> tuple[str, str] | None:
    func_name = _call_name(node.func)
    if func_name == "asset":
        return _constant_pair(node.args[:2])
    if func_name == "register":
        return _constant_pair(node.args[:2])
    return None


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _constant_pair(args: Iterable[ast.expr]) -> tuple[str, str] | None:
    values = list(args)
    if len(values) < 2:
        return None
    left = _string_value(values[0])
    right = _string_value(values[1])
    if not left or not right:
        return None
    return left, right


def _string_value(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        value = str(node.value).strip()
        return value or None
    return None


def _declared_in(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _module_name_for_path(path: Path) -> str:
    relative = path.relative_to(SRC_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _iter_python_paths() -> Iterable[Path]:
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def collect_assets() -> dict[tuple[str, str], AssetRecord]:
    records: dict[tuple[str, str], AssetRecord] = {}
    for path in _iter_python_paths():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                warnings.simplefilter("ignore", DeprecationWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        module_name = _module_name_for_path(path)
        collector = AssetCollector(path=path, module_name=module_name)
        collector.visit(tree)
        for record in collector.records:
            if not _module_exists(record.module):
                raise RuntimeError(f"Unable to resolve module '{record.module}' for {path}")
            records[(record.kind, record.name)] = record
    return records


def _module_exists(module_name: str) -> bool:
    parts = [part for part in str(module_name).split(".") if part]
    if not parts or parts[0] != "gage_eval":
        return False
    module_file = SRC_ROOT.joinpath(*parts).with_suffix(".py")
    package_init = SRC_ROOT.joinpath(*parts, "__init__.py")
    return module_file.is_file() or package_init.is_file()


def build_manifest_payloads() -> dict[str, list[dict[str, object]]]:
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    records = collect_assets()
    shards: dict[str, list[dict[str, object]]] = {
        "core": [],
        "datasets": [],
        "arena": [],
        "backends": [],
        "metrics": [],
        "manual_overrides": [dict(entry) for entry in MANUAL_OVERRIDES],
    }
    for record in sorted(records.values(), key=lambda item: (item.kind, item.name, item.module)):
        shard = SHARD_BY_KIND.get(record.kind, DEFAULT_SHARD)
        shards.setdefault(shard, []).append(record.to_payload())
    return shards


def write_manifests(*, check: bool) -> int:
    payloads = build_manifest_payloads()
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    (MANIFEST_ROOT / "plugins").mkdir(parents=True, exist_ok=True)
    expected = {
        "core.json": payloads["core"],
        "datasets.json": payloads["datasets"],
        "arena.json": payloads["arena"],
        "backends.json": payloads["backends"],
        "metrics.json": payloads["metrics"],
        "manual_overrides.json": payloads["manual_overrides"],
    }
    dirty = False
    for filename, entries in expected.items():
        path = MANIFEST_ROOT / filename
        content = json.dumps(
            {"manifest_version": MANIFEST_VERSION, "entries": entries},
            ensure_ascii=True,
            indent=2,
            sort_keys=False,
        ) + "\n"
        previous = path.read_text(encoding="utf-8") if path.exists() else None
        if previous != content:
            dirty = True
            if not check:
                path.write_text(content, encoding="utf-8")
    return 1 if check and dirty else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate manifests without rewriting them")
    args = parser.parse_args()
    return write_manifests(check=bool(args.check))


if __name__ == "__main__":
    raise SystemExit(main())
