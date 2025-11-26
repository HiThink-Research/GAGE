"""JSONL dataset loader used by the DataManager."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.assets.datasets.hubs.base import DatasetHubHandle
from gage_eval.assets.datasets.loaders.base import DatasetLoader
from gage_eval.assets.datasets.manager import DataSource
from gage_eval.assets.datasets.loaders.loader_utils import (
    apply_default_params,
    apply_preprocess,
    resolve_doc_to_callable,
)
from gage_eval.registry import registry

_STREAMING_THRESHOLD_MB = 512


@registry.asset(
    "dataset_loaders",
    "jsonl",
    desc="本地 JSONL 文件加载器",
    tags=("local", "file"),
    supports_streaming=True,
)
class JSONLDatasetLoader(DatasetLoader):
    """Build a :class:`DataSource` from a JSONL file."""

    def load(self, hub_handle: Optional[DatasetHubHandle]) -> DataSource:
        path = _resolve_path(self.spec, hub_handle)
        if not path:
            raise ValueError(f"Dataset '{self.spec.dataset_id}' missing JSONL 'path' argument")
        filesystem_path = Path(path).expanduser()
        if not filesystem_path.exists():
            raise FileNotFoundError(f"Dataset '{self.spec.dataset_id}' JSONL file not found: {filesystem_path}")

        limit = self.spec.params.get("limit")
        streaming = _should_stream_jsonl(self.spec, filesystem_path, hub_handle)
        raw_records: Iterable[Dict[str, Any]]
        if streaming:
            raw_records = _stream_jsonl(filesystem_path, limit=limit)
        else:
            raw_records = _read_jsonl(filesystem_path, limit=limit)

        records = apply_preprocess(raw_records, self.spec, data_path=str(filesystem_path))
        records = apply_default_params(records, self.spec)
        if not streaming:
            records = list(records)

        doc_to_text = resolve_doc_to_callable(self.spec, "doc_to_text")
        doc_to_visual = resolve_doc_to_callable(self.spec, "doc_to_visual")
        doc_to_audio = resolve_doc_to_callable(self.spec, "doc_to_audio")
        metadata = {
            "loader": "jsonl",
            "path": str(filesystem_path),
            "split": self.spec.params.get("split", "unspecified"),
            "subset": self.spec.params.get("subset"),
            "streaming": streaming,
        }

        return DataSource(
            dataset_id=self.spec.dataset_id,
            records=records,
            doc_to_text=doc_to_text,
            doc_to_visual=doc_to_visual,
            doc_to_audio=doc_to_audio,
            metadata=metadata,
            validation=self.spec.schema,
            streaming=streaming,
        )


def _resolve_path(spec: DatasetSpec, hub_handle: Optional[DatasetHubHandle]) -> Optional[str]:
    if hub_handle:
        if hub_handle.metadata.get("path"):
            return str(hub_handle.metadata["path"])
        if isinstance(hub_handle.resource, str):
            return hub_handle.resource
    return spec.params.get("path")


def _read_jsonl(path: Path, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    data: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                data.append(json.loads(line))
            if limit is not None and index + 1 >= limit:
                break
    return data


def _stream_jsonl(path: Path, limit: Optional[int]) -> Iterator[Dict[str, Any]]:
    def generator():
        with path.open("r", encoding="utf-8") as handle:
            yielded = 0
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)
                yielded += 1
                if limit is not None and yielded >= limit:
                    break

    return generator()


def _should_stream_jsonl(
    spec: DatasetSpec,
    path: Path,
    hub_handle: Optional[DatasetHubHandle],
) -> bool:
    if hub_handle and hub_handle.streaming:
        return True
    flag = spec.params.get("streaming")
    if flag is not None:
        return bool(flag)
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0
    threshold_mb = float(spec.params.get("streaming_threshold_mb", _STREAMING_THRESHOLD_MB))
    if size_bytes >= threshold_mb * 1024 * 1024:
        if spec.params.get("limit") in (None, 0):
            return True
    return False
