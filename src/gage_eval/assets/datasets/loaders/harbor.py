"""Descriptor-only dataset loaders for Harbor-backed external harness runs."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.assets.datasets.hubs.base import DatasetHubHandle
from gage_eval.assets.datasets.loaders.base import DatasetLoader
from gage_eval.assets.datasets.manager import DataSource
from gage_eval.registry import registry

HARBOR_DATASET_METADATA_KEY = "external_harness_dataset"


@registry.asset(
    "dataset_loaders",
    "harbor_registry",
    desc="Descriptor-only Harbor registry dataset loader",
    tags=("external_harness", "harbor"),
)
class HarborRegistryDatasetLoader(DatasetLoader):
    """Return Harbor registry metadata without materializing sample records."""

    def load(self, hub_handle: Optional[DatasetHubHandle], *, trace=None) -> DataSource:
        params = _merge_params(self.spec.params, hub_handle)
        dataset_config = _compact(
            {
                "ref": params.get("ref"),
                "name": params.get("name"),
                "version": params.get("version"),
                "registry_path": params.get("registry_path"),
                "registry_url": params.get("registry_url"),
                "n_tasks": params.get("n_tasks") or params.get("max_samples"),
                "task_filter": params.get("task_filter"),
                "task_names": params.get("task_names"),
                "exclude_task_names": params.get("exclude_task_names"),
            }
        )
        return _descriptor_source(
            dataset_id=self.spec.dataset_id,
            loader="harbor_registry",
            harbor_metadata={
                "source": "registry",
                "dataset_config": dataset_config,
                "params": dict(params),
            },
        )


@registry.asset(
    "dataset_loaders",
    "harbor_local_path",
    desc="Descriptor-only Harbor local path dataset loader",
    tags=("external_harness", "harbor"),
)
class HarborLocalPathDatasetLoader(DatasetLoader):
    """Return Harbor local task/dataset path metadata without scanning it."""

    def load(self, hub_handle: Optional[DatasetHubHandle], *, trace=None) -> DataSource:
        params = _merge_params(self.spec.params, hub_handle)
        path = params.get("path")
        local_path = _compact(
            {
                "path": path,
                "task_path": path,
                "task_name": params.get("task_name"),
                "path_kind": params.get("path_kind", "auto"),
                "path_scope": params.get("path_scope", "host"),
                "task_names": params.get("task_names"),
                "exclude_task_names": params.get("exclude_task_names"),
            }
        )
        return _descriptor_source(
            dataset_id=self.spec.dataset_id,
            loader="harbor_local_path",
            harbor_metadata={
                "source": "local_path",
                "local_path": local_path,
                "params": dict(params),
            },
        )


def _descriptor_source(
    *,
    dataset_id: str,
    loader: str,
    harbor_metadata: Dict[str, Any],
) -> DataSource:
    return DataSource(
        dataset_id=dataset_id,
        records=(),
        metadata={
            "loader": loader,
            HARBOR_DATASET_METADATA_KEY: True,
            "harbor": harbor_metadata,
        },
        streaming=False,
    )


def _merge_params(
    params: Dict[str, Any],
    hub_handle: Optional[DatasetHubHandle],
) -> Dict[str, Any]:
    merged = dict(params or {})
    if hub_handle is not None:
        merged = {**dict(hub_handle.metadata or {}), **merged}
        if hub_handle.resource is not None and "path" not in merged:
            merged["path"] = hub_handle.resource
    return merged


def _compact(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


__all__ = [
    "HARBOR_DATASET_METADATA_KEY",
    "HarborLocalPathDatasetLoader",
    "HarborRegistryDatasetLoader",
]
