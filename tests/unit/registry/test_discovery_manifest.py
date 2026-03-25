from __future__ import annotations

import json
from pathlib import Path

import pytest

from gage_eval.registry import (
    RegistryRuntimeMutationError,
    load_default_manifest_repository,
    registry,
)
from gage_eval.registry.discovery_manifest import DiscoveryManifestRepository


@pytest.mark.fast
def test_default_manifest_repository_contains_manual_prompt_overrides() -> None:
    repository = load_default_manifest_repository()

    entry = repository.require("prompts", "dut/general@v1")

    assert entry.module == "gage_eval.assets.prompts.catalog"
    assert entry.load_phase == "baseline"


@pytest.mark.fast
def test_manifest_repository_merges_manual_overrides_last(tmp_path: Path) -> None:
    root = tmp_path / "manifests"
    root.mkdir(parents=True, exist_ok=True)
    (root / "plugins").mkdir(parents=True, exist_ok=True)
    (root / "core.json").write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "entries": [
                    {
                        "kind": "roles",
                        "name": "demo_role",
                        "module": "demo.core",
                        "load_phase": "prepare_only",
                        "declared_in": "src/demo/core.py",
                        "aliases": [],
                        "optional": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "manual_overrides.json").write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "entries": [
                    {
                        "kind": "roles",
                        "name": "demo_role",
                        "module": "demo.override",
                        "load_phase": "baseline",
                        "declared_in": "src/demo/override.py",
                        "aliases": ["demo_role_alias"],
                        "optional": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    repository = DiscoveryManifestRepository.from_roots((root,))

    assert repository.require("roles", "demo_role").module == "demo.override"
    assert repository.require("roles", "demo_role_alias").module == "demo.override"


@pytest.mark.fast
def test_frozen_registry_view_blocks_runtime_mutation() -> None:
    clone = registry.clone()
    view = clone.freeze(view_id="unit-view")

    with pytest.raises(RegistryRuntimeMutationError):
        view.register("roles", "late", object(), desc="late")

    with pytest.raises(RegistryRuntimeMutationError):
        view.auto_discover("roles", "gage_eval.role.adapters")

    with pytest.raises(RegistryRuntimeMutationError):
        view.extra_attr = "boom"
