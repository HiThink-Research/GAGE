"""ARC-AGI-2 dataset loader for directory of JSON files."""

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


@registry.asset(
    "dataset_loaders",
    "arcagi2",
    desc="ARC-AGI-2 dataset loader (directory of JSON files)",
    tags=("local", "file"),
    supports_streaming=False,
)
class ARCAGI2DatasetLoader(DatasetLoader):
    """Build a :class:`DataSource` from a directory of ARC-AGI-2 JSON files."""

    def load(self, hub_handle: Optional[DatasetHubHandle], *, trace=None) -> DataSource:
        path = _resolve_path(self.spec, hub_handle)
        if not path:
            raise ValueError(f"Dataset '{self.spec.dataset_id}' missing ARC-AGI-2 'path' argument")

        filesystem_path = Path(path).expanduser().resolve()
        if not filesystem_path.exists():
            message = f"Dataset '{self.spec.dataset_id}' ARC-AGI-2 directory not found: {filesystem_path}"
            raise FileNotFoundError(message)

        if not filesystem_path.is_dir():
            raise ValueError(f"Dataset '{self.spec.dataset_id}' ARC-AGI-2 path must be a directory: {filesystem_path}")

        limit = self.spec.params.get("limit")
        raw_records = _read_arcagi2_directory(filesystem_path, limit=limit)

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

        # Count total files in directory
        total_files = len(list(filesystem_path.glob('*.json')))

        metadata = {
            "loader": "arcagi2",
            "path": str(filesystem_path),
            "split": "evaluation",  # ARC-AGI-2 uses "evaluation" split
            "streaming": False,
            "total_files": total_files,
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


def _resolve_path(spec: DatasetSpec, hub_handle: Optional[DatasetHubHandle]) -> Optional[str]:
    if hub_handle:
        if hub_handle.metadata.get("path"):
            return str(hub_handle.metadata["path"])
        if isinstance(hub_handle.resource, str):
            return hub_handle.resource
    return spec.params.get("path")


def _read_arcagi2_directory(path: Path, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    """Read all JSON files from ARC-AGI-2 directory."""
    data: list[Dict[str, Any]] = []

    # Get all JSON files in directory, sorted for reproducibility
    json_files = sorted(path.glob('*.json'))

    for json_file in json_files:
        try:
            with json_file.open('r', encoding='utf-8') as f:
                record = json.load(f)

            # Add filename as problem_id if not present
            if 'id' not in record and 'task_id' not in record:
                record['id'] = json_file.stem

            data.append(record)

            if limit is not None and len(data) >= limit:
                break

        except (json.JSONDecodeError, IOError) as e:
            # Skip malformed files
            print(f"Warning: Skipping malformed file {json_file}: {e}")
            continue

    return data