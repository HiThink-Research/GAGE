from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "registry" / "sync_registry_manifest.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("sync_registry_manifest", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.fast
def test_manifest_sync_script_reports_clean_tree_after_generation() -> None:
    module = _load_script_module()

    assert module.write_manifests(check=True) == 0


@pytest.mark.fast
def test_manifest_sync_payloads_cover_known_assets() -> None:
    module = _load_script_module()
    payloads = module.build_manifest_payloads()

    core_names = {(entry["kind"], entry["name"]) for entry in payloads["core"]}
    arena_names = {(entry["kind"], entry["name"]) for entry in payloads["arena"]}
    metric_names = {(entry["kind"], entry["name"]) for entry in payloads["metrics"]}
    override_names = {(entry["kind"], entry["name"]) for entry in payloads["manual_overrides"]}

    assert ("roles", "dut_model") in core_names
    assert ("arena_impls", "gomoku_local_v1") in arena_names
    assert ("visualization_specs", "arena/visualization/gomoku_board_v1") in arena_names
    assert ("visualization_specs", "arena/visualization/doudizhu_table_v1") in arena_names
    assert ("visualization_specs", "arena/visualization/vizdoom_frame_v1") in arena_names
    assert ("metrics", "global_piqa_accuracy_local") in metric_names
    assert ("prompts", "dut/general@v1") in override_names
