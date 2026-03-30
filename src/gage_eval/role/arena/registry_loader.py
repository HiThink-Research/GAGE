"""Manifest-backed imports for arena assets."""

from __future__ import annotations

from gage_eval.registry import import_asset_from_manifest, import_kind_from_manifest, registry


def import_arena_asset_module(kind: str, asset_name: str, *, registry_lookup=registry, source: str = ""):
    return import_asset_from_manifest(
        str(kind),
        str(asset_name),
        registry=registry_lookup,
        source=source or f"arena_loader:{kind}",
    )


def import_all_arena_asset_modules(kind: str, *, registry_lookup=registry):
    return import_kind_from_manifest(str(kind), registry=registry_lookup)
