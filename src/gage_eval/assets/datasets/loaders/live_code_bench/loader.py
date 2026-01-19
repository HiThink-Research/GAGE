"""Loader capable of pulling Live Code Bench datasets from HuggingFace Hub."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional
from dataclasses import dataclass, asdict

from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.assets.datasets.hubs.base import DatasetHubHandle
from gage_eval.assets.datasets.loaders.base import DatasetLoader
from gage_eval.assets.datasets.manager import DataSource
from gage_eval.assets.datasets.loaders.hf_hub_loader import _maybe_apply_preprocess

from gage_eval.assets.datasets.loaders.loader_utils import (
    apply_default_params,
    apply_bundle,
    apply_preprocess,
    build_bundle_context,
    build_preprocess_context,
    inject_default_params,
    resolve_doc_to_callable,
    )
from gage_eval.assets.datasets.loaders.live_code_bench.utils import (
    ALLOWED_FILES,
    get_filename_list
)
from gage_eval.assets.datasets.loaders.live_code_bench.code_generation import (
    CodeGenerationProblem, 
    load_code_generation_dataset_not_fast, 
    load_code_generation_dataset   
)
from gage_eval.assets.datasets.loaders.live_code_bench.scenarios import (
    Scenario, 
)

from gage_eval.observability.config import get_observability_config
from gage_eval.registry import registry
import logging
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError # Added import

logger = logging.getLogger(__name__)


@registry.asset(
    "dataset_loaders",
    "hf_hub_live_code_bench", # hf_hub_live_code_bench
    desc="[live code benchRemote] dataset loader for HuggingFace Hub ",
    tags=("remote", "huggingface"),
    supports_streaming=True,
)
class HuggingFaceDatasetLoader(DatasetLoader):
    def load(self, hub_handle: Optional[DatasetHubHandle], *, trace=None) -> DataSource:
        return load_live_code_bench_hf_hub_dataset(self.spec, hub_handle, trace=trace)


def _normalize_source(raw: str) -> str:
    normalized = raw.lower()
    if normalized not in {"huggingface", "modelscope", "local"}:
        raise ValueError(f"Unsupported dataset source: {raw}")
    return normalized

def _apply_limit(dataset, limit):
    if isinstance(limit, float):
        if not 0 < limit <= 1:
            raise ValueError("Float limit must be between 0 and 1")
        size = int(len(dataset) * limit)
        return dataset[:size]
    if isinstance(limit, int):
        if limit < 0:
            raise ValueError("Limit must be non-negative")
    return dataset[:limit]


def load_live_code_bench_hf_hub_dataset(spec: DatasetSpec, hub_handle: Optional[DatasetHubHandle] = None, *, trace=None) -> DataSource:
    """Download/cache a remote dataset and expose it as a DataSource."""

    try:
        import datasets  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "`datasets` package is required to load HuggingFace/ModelScope datasets"
        ) from exc

    metadata = dict(hub_handle.metadata) if hub_handle and hub_handle.metadata else {}
    loader_params = dict(metadata)
    loader_params.update(spec.params)
    streaming_override = None if hub_handle is None else hub_handle.streaming

    hub_id = loader_params.get("hub_id") or loader_params.get("path")
    if not hub_id:
        raise ValueError(f"Dataset '{spec.dataset_id}' missing 'hub_id' argument")

    default_source = "modelscope" if spec.loader == "modelscope" else "huggingface"
    source = _normalize_source(loader_params.get("source", default_source))
    split = loader_params.get("split", "train")
    subset = loader_params.get("subset")
    revision = loader_params.get("revision")
    trust_remote = loader_params.get("trust_remote_code", True)
    load_kwargs = dict(loader_params.get("load_kwargs", {}))
    builder_name = loader_params.get("builder_name")
    data_files = loader_params.get("data_files")
    local_dir = loader_params.get("local_dir")
    subset_for_metadata = subset
    streaming = False
    if data_files:
        streaming = False  # Local file mode does not support streaming.
    cache_path = None

    logger.info(
        "Loading HF dataset hub_id={} subset={} split={} streaming={} cache={}",
        hub_id,
        subset_for_metadata or "default",
        split,
        streaming_override if streaming_override is not None else False,
        "auto",
    )
    print("in dataloader")
    local_dir = snapshot_download(
        repo_id = hub_id,
        repo_type = "dataset",
        local_dir=local_dir
    )
    print(f'saved to local_dir: {local_dir}')

    scenario = loader_params.get("scenario")    
    not_fast = loader_params.get("not_fast")
    start_date = str(loader_params.get("start_date"))
    end_date = str(loader_params.get("end_date"))
    release_version = loader_params.get("release_version")
    print(scenario, not_fast, start_date, end_date, release_version)
    filename_list  = get_filename_list(release_version)
    if scenario == Scenario.codegeneration.value:
        if not_fast:
            benchmark = load_code_generation_dataset_not_fast(local_dir, filename_list)
        else:
            benchmark = load_code_generation_dataset(local_dir, filename_list, start_date, end_date)
    else:
        print(f"not supported: {scenario}")
    
    limit = loader_params.get("limit")
    if limit is not None:
        benchmark = _apply_limit(benchmark, limit)

    def _add_scenario(sample):
        ret = asdict(sample)
        ret['scenario'] = scenario
        return ret
        
    dataset = map(_add_scenario, benchmark)

    records = _maybe_apply_preprocess(
            dataset,
            spec,
            data_path=hub_id,
            trace=trace,
    )

    metadata_payload = {
        "loader": "hf_hub",
        "hub_id": hub_id,
        "source": source,
        "split": split,
        "subset": subset_for_metadata or "default",
        "revision": revision or "latest",
        "cache_path": cache_path,
        "streaming": streaming,
    }
    for key, value in metadata.items():
        if key not in metadata_payload and value is not None:
            metadata_payload[key] = value
    return DataSource(
        dataset_id=spec.dataset_id,
        records=records,
        doc_to_text=None,
        doc_to_visual=None,
        doc_to_audio=None,
        metadata=metadata_payload,
        validation=spec.schema,
        streaming=streaming,
    )


