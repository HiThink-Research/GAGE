"""Manifest-backed imports for dataset-related assets."""

from __future__ import annotations

from gage_eval.registry import import_asset_from_manifest, registry


def import_dataset_asset_module(kind: str, asset_name: str) -> None:
    import_asset_from_manifest(str(kind), str(asset_name), registry=registry, source=f"dataset_loader:{kind}")
