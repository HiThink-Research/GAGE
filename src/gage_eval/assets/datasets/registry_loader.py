"""Manifest-backed imports for dataset-related assets."""

from __future__ import annotations

from typing import Any

from gage_eval.registry import import_asset_from_manifest, registry


def import_dataset_asset_module(kind: str, asset_name: str, *, registry_lookup: Any = None) -> None:
    target_registry = registry_lookup or registry
    import_asset_from_manifest(
        str(kind),
        str(asset_name),
        registry=target_registry,
        source=f"dataset_loader:{kind}",
    )
