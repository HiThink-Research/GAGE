"""CSV dataset loader used by the DataManager."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

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


def _read_csv(path: Path, *, limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            yield dict(row)
            if limit is not None and idx + 1 >= limit:
                break


def _stream_csv(path: Path, *, limit: Optional[int]) -> Iterator[Dict[str, Any]]:
    def generator():
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            yielded = 0
            for row in reader:
                yield dict(row)
                yielded += 1
                if limit is not None and yielded >= limit:
                    break

    return generator()


@registry.asset(
    "dataset_loaders",
    "csv",
    desc="Local CSV file dataset loader",
    tags=("local", "file"),
    supports_streaming=True,
)
class CSVDatasetLoader(DatasetLoader):
    """Build a :class:`DataSource` from a CSV file."""

    def load(self, hub_handle: Optional[DatasetHubHandle], *, trace=None) -> DataSource:
        path = _resolve_path(self.spec, hub_handle)
        if not path:
            raise ValueError(f"Dataset '{self.spec.dataset_id}' missing CSV 'path' argument")
        filesystem_path = Path(path).expanduser()
        if not filesystem_path.exists():
            raise FileNotFoundError(f"Dataset '{self.spec.dataset_id}' CSV file not found: {filesystem_path}")

        limit = self.spec.params.get("limit")
        streaming = bool(self.spec.params.get("streaming"))
        if streaming:
            raw_records = _stream_csv(filesystem_path, limit=limit)
        else:
            raw_records = _read_csv(filesystem_path, limit=limit)

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
        if not streaming:
            records = list(records)
        metadata = {
            "loader": "csv",
            "path": str(filesystem_path),
            "split": self.spec.params.get("split", "unspecified"),
            "subset": self.spec.params.get("subset"),
            "streaming": streaming,
        }

        return DataSource(
            dataset_id=self.spec.dataset_id,
            records=records,
            doc_to_text=None,
            doc_to_visual=None,
            doc_to_audio=None,
            metadata=metadata,
            validation=self.spec.schema,
            streaming=streaming,
        )
