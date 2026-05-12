from __future__ import annotations

from pathlib import Path

import pytest

from gage_eval.assets.datasets.manager import DataManager
from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.registry import (
    clear_manifest_repository_cache,
    load_default_manifest_repository,
)


@pytest.mark.fast
def test_harbor_registry_descriptor_preserves_flat_dataset_config_fields() -> None:
    from gage_eval.assets.datasets.loaders.harbor import HarborRegistryDatasetLoader

    spec = DatasetSpec(
        dataset_id="tb2_registry",
        loader="harbor_registry",
        params={
            "name": "terminal-bench",
            "version": "2.0",
            "registry_path": "harbor-registry",
            "n_tasks": 1,
            "task_filter": {"task_ids": ["gpt2-codegolf"]},
        },
    )

    source = HarborRegistryDatasetLoader(spec).load(None)

    assert tuple(source.records) == ()
    assert source.metadata["loader"] == "harbor_registry"
    assert source.metadata["external_harness_dataset"] is True
    assert source.metadata["harbor"]["dataset_config"] == {
        "name": "terminal-bench",
        "version": "2.0",
        "registry_path": "harbor-registry",
        "n_tasks": 1,
        "task_filter": {"task_ids": ["gpt2-codegolf"]},
    }


@pytest.mark.fast
def test_harbor_local_path_descriptor_preserves_task_path_and_name() -> None:
    from gage_eval.assets.datasets.loaders.harbor import HarborLocalPathDatasetLoader

    spec = DatasetSpec(
        dataset_id="tb2_local_case",
        loader="harbor_local_path",
        params={
            "path": "/Users/panke/.cache/harbor/tasks/unit/gpt2-codegolf",
            "task_name": "gpt2-codegolf",
            "path_kind": "task",
            "path_scope": "host",
        },
    )

    source = HarborLocalPathDatasetLoader(spec).load(None)

    assert tuple(source.records) == ()
    assert source.metadata["loader"] == "harbor_local_path"
    assert source.metadata["external_harness_dataset"] is True
    assert source.metadata["harbor"]["local_path"] == {
        "path": "/Users/panke/.cache/harbor/tasks/unit/gpt2-codegolf",
        "task_path": "/Users/panke/.cache/harbor/tasks/unit/gpt2-codegolf",
        "task_name": "gpt2-codegolf",
        "path_kind": "task",
        "path_scope": "host",
    }


@pytest.mark.fast
def test_harbor_local_path_loader_does_not_scan_task_directory_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gage_eval.assets.datasets.loaders.harbor import HarborLocalPathDatasetLoader

    def fail_scan(*args, **kwargs):
        raise AssertionError("harbor descriptor loader must not scan local task contents")

    monkeypatch.setattr(Path, "iterdir", fail_scan)
    monkeypatch.setattr(Path, "glob", fail_scan)
    monkeypatch.setattr(Path, "rglob", fail_scan)
    spec = DatasetSpec(
        dataset_id="tb2_local_case",
        loader="harbor_local_path",
        params={"path": str(tmp_path / "not-scanned"), "path_kind": "auto"},
    )

    source = HarborLocalPathDatasetLoader(spec).load(None)

    assert source.metadata["harbor"]["local_path"]["path"] == str(tmp_path / "not-scanned")


@pytest.mark.fast
def test_sample_loop_iteration_rejects_harbor_dataset_descriptor() -> None:
    from gage_eval.assets.datasets.loaders.harbor import HarborRegistryDatasetLoader

    source = HarborRegistryDatasetLoader(
        DatasetSpec(
            dataset_id="tb2_registry",
            loader="harbor_registry",
            params={"ref": "terminal-bench@2.0"},
        )
    ).load(None)
    manager = DataManager()
    manager.register_source(source)

    with pytest.raises(ValueError, match="external_harness.config.invalid_dataset_params"):
        list(manager.iter_samples("tb2_registry"))


@pytest.mark.fast
def test_harbor_dataset_loaders_are_manifest_discoverable() -> None:
    clear_manifest_repository_cache()
    repository = load_default_manifest_repository()

    registry_entry = repository.require("dataset_loaders", "harbor_registry")
    local_entry = repository.require("dataset_loaders", "harbor_local_path")

    assert registry_entry.module == "gage_eval.assets.datasets.loaders.harbor"
    assert local_entry.module == "gage_eval.assets.datasets.loaders.harbor"
