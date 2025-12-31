"""Parquet dataset loader used by the DataManager."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.assets.datasets.hubs.base import DatasetHubHandle
from gage_eval.assets.datasets.loaders.base import DatasetLoader
from gage_eval.assets.datasets.loaders.loader_utils import (
    apply_default_params,
    apply_preprocess,
    resolve_doc_to_callable,
)
from gage_eval.assets.datasets.manager import DataSource
from gage_eval.registry import registry


def _resolve_path(spec: DatasetSpec, hub_handle: Optional[DatasetHubHandle]) -> Optional[str]:
    if hub_handle:
        if hub_handle.metadata.get("path"):
            return str(hub_handle.metadata["path"])
        if isinstance(hub_handle.resource, str):
            return hub_handle.resource
    return spec.params.get("path")


def _read_parquet(path: Path, *, limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pyarrow is required to load parquet datasets") from exc

    table = pq.read_table(path)
    if limit is not None:
        table = table.slice(0, limit)
    return table.to_pylist()


@registry.asset(
    "dataset_loaders",
    "parquet",
    desc="本地 Parquet 文件加载器",
    tags=("local", "file"),
    supports_streaming=False,
)
class ParquetDatasetLoader(DatasetLoader):
    """Build a :class:`DataSource` from a Parquet file."""

    def load(self, hub_handle: Optional[DatasetHubHandle], *, trace=None) -> DataSource:
        path = _resolve_path(self.spec, hub_handle)
        if not path:
            raise ValueError(f"Dataset '{self.spec.dataset_id}' missing Parquet 'path' argument")
        filesystem_path = Path(path).expanduser()
        if not filesystem_path.exists():
            raise FileNotFoundError(f"Dataset '{self.spec.dataset_id}' Parquet file not found: {filesystem_path}")

        limit = self.spec.params.get("limit")
        raw_records = _read_parquet(filesystem_path, limit=limit)
        doc_to_text = resolve_doc_to_callable(self.spec, "doc_to_text")
        doc_to_visual = resolve_doc_to_callable(self.spec, "doc_to_visual")
        doc_to_audio = resolve_doc_to_callable(self.spec, "doc_to_audio")
        records = apply_preprocess(
            raw_records,
            self.spec,
            data_path=str(filesystem_path),
            doc_to_text=doc_to_text,
            doc_to_visual=doc_to_visual,
            doc_to_audio=doc_to_audio,
            trace=trace,
        )
        records = apply_default_params(records, self.spec)
        records = list(records)
        metadata = {
            "loader": "parquet",
            "path": str(filesystem_path),
            "split": self.spec.params.get("split", "unspecified"),
            "subset": self.spec.params.get("subset"),
            "streaming": False,
        }

        return DataSource(
            dataset_id=self.spec.dataset_id,
            records=records,
            doc_to_text=None,
            doc_to_visual=None,
            doc_to_audio=None,
            metadata=metadata,
            validation=self.spec.schema,
            streaming=False,
        )
