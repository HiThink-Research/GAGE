from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest

from gage_eval.registry import RegistryBootstrapCoordinator
from gage_eval.registry.asset_planner import DiscoveryPlan, DiscoveryRequest
from gage_eval.registry.bootstrap_importer import BootstrapImporter
from gage_eval.registry.discovery_manifest import (
    DiscoveryManifestRepository,
    clear_manifest_repository_cache,
)
from gage_eval.registry.manager import RegistryManager


def _write_manifest(root: Path, entries: list[dict[str, object]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "core.json").write_text(
        json.dumps({"manifest_version": 1, "entries": entries}),
        encoding="utf-8",
    )


@pytest.mark.fast
def test_bootstrap_importer_executes_baseline_requests_before_prepare_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_manifest(
        tmp_path,
        [
            {
                "kind": "roles",
                "name": "prepare_role",
                "module": "demo.prepare",
                "load_phase": "prepare_only",
                "declared_in": "src/demo/prepare.py",
            },
            {
                "kind": "roles",
                "name": "baseline_role",
                "module": "demo.baseline",
                "load_phase": "baseline",
                "declared_in": "src/demo/baseline.py",
            },
        ],
    )
    registry = RegistryManager()
    registry.declare_kind("roles", desc="roles")
    importer = BootstrapImporter(DiscoveryManifestRepository.from_roots((tmp_path,)))
    imported_modules: list[str] = []

    def _registering_import(module_name: str) -> ModuleType:
        imported_modules.append(module_name)
        if module_name == "demo.prepare":
            registry.register("roles", "prepare_role", object(), desc="prepare role")
        elif module_name == "demo.baseline":
            registry.register("roles", "baseline_role", object(), desc="baseline role")
        return ModuleType(module_name)

    monkeypatch.setattr("gage_eval.registry.bootstrap_importer.importlib.import_module", _registering_import)
    monkeypatch.setattr("gage_eval.registry.bootstrap_importer.importlib.reload", lambda module: module)

    report = importer.execute(
        DiscoveryPlan(
            requests=(
                DiscoveryRequest(kind="roles", name="prepare_role", source="prepare"),
                DiscoveryRequest(kind="roles", name="baseline_role", source="baseline"),
            )
        ),
        registry=registry,
    )

    assert report.ok is True
    assert imported_modules == ["demo.baseline", "demo.prepare"]
    assert tuple(item.module for item in report.imported) == ("demo.baseline", "demo.prepare")


@pytest.mark.fast
def test_registry_bootstrap_coordinator_refreshes_default_manifest_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_v1 = tmp_path / "v1"
    root_v2 = tmp_path / "v2"
    _write_manifest(
        root_v1,
        [
            {
                "kind": "roles",
                "name": "demo_role",
                "module": "demo.v1",
                "load_phase": "baseline",
                "declared_in": "src/demo/v1.py",
            }
        ],
    )
    _write_manifest(
        root_v2,
        [
            {
                "kind": "roles",
                "name": "demo_role",
                "module": "demo.v2",
                "load_phase": "baseline",
                "declared_in": "src/demo/v2.py",
            }
        ],
    )
    clear_manifest_repository_cache()
    monkeypatch.setenv("GAGE_EVAL_DISCOVERY_MANIFEST_ROOTS", str(root_v1))
    registry = RegistryManager()
    registry.declare_kind("roles", desc="roles")
    coordinator = RegistryBootstrapCoordinator(registry)

    imported_modules: list[str] = []

    def _registering_import(module_name: str) -> ModuleType:
        imported_modules.append(module_name)
        registry.register("roles", "demo_role", object(), desc=f"{module_name} role")
        return ModuleType(module_name)

    monkeypatch.setattr("gage_eval.registry.bootstrap_importer.importlib.import_module", _registering_import)
    monkeypatch.setattr("gage_eval.registry.bootstrap_importer.importlib.reload", lambda module: module)

    first = coordinator.prepare_runtime_context(
        run_id="run-v1",
        discovery_plan=DiscoveryPlan(
            requests=(DiscoveryRequest(kind="roles", name="demo_role", source="unit"),)
        ),
    )
    try:
        assert tuple(item.module for item in first.discovery_report.imported) == ("demo.v1",)
    finally:
        first.close()

    monkeypatch.setenv("GAGE_EVAL_DISCOVERY_MANIFEST_ROOTS", str(root_v2))
    clear_manifest_repository_cache()
    second = coordinator.prepare_runtime_context(
        run_id="run-v2",
        discovery_plan=DiscoveryPlan(
            requests=(DiscoveryRequest(kind="roles", name="demo_role", source="unit"),)
        ),
    )
    try:
        assert tuple(item.module for item in second.discovery_report.imported) == ("demo.v2",)
    finally:
        second.close()

    assert imported_modules == ["demo.v1", "demo.v2"]
