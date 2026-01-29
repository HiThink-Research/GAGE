"""Loader capable of pulling BizFin Bench V2 datasets from HuggingFace Hub."""

from __future__ import annotations

import hashlib
import json
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer as Loader  # type: ignore


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

from gage_eval.assets.datasets.loaders.biz_fin_bench.utils import (
    get_file_path,
)

from gage_eval.observability.config import get_observability_config
from gage_eval.registry import registry
import logging
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError # Added import
import tiktoken
import json
import pandas as pd
from datasets import Dataset
logger = logging.getLogger(__name__)

def get_needle_filename_list(needle_type):
    ret = []
    for i in range(2):
        ret.append(
            f"{needle_type}/{needle_type}_{i}.parquet"
        )
    return ret

def global_filter(example, max_content_window,
                   tokenizer_type,
                    tokenizer_name_or_path):
    if tokenizer_type == 'local':
        tokenizer = Loader.from_pretrained(tokenizer_name_or_path)
        messages = json.loads(example["prompt"])
        chat_template = getattr(tokenizer, 'chat_template', None)
        if chat_template:
            input = tokenizer.apply_chat_template(
            messages, tokenize=True, 
            add_generation_prompt=False
            )
        else:
            input = tokenizer.encode(messages, add_special_tokens=False)
        input_len = len(input)
    else:
        enc = tiktoken.get_encoding(tokenizer_name_or_path)
        def n_tokens(messages : list[dict]) -> int:
            """
            Count tokens in messages.
            """
            return sum([len(enc.encode(m["content"])) for m in messages])
        messages = json.loads(example["prompt"])
        input_len = n_tokens(messages)
    return input_len <= max_content_window

def _load_data(filename_list):
    dataset = pd.concat([
        pd.read_parquet(
        hf_hub_download(repo_id="openai/mrcr",
        filename=filename, repo_type="dataset")) for filename in filename_list])
    return dataset

@registry.asset(
    "dataset_loaders",
    "openai_mrcr_loader", # hf_hub_live_code_bench
    desc="[openai mrcr]] dataset loader for HuggingFace Hub ",
    tags=("remote", "huggingface"),
    supports_streaming=True,
)
class HuggingFaceDatasetLoader(DatasetLoader):
    def load(self, hub_handle: Optional[DatasetHubHandle], *, trace=None) -> DataSource:
        return openai_mrcr_loader_hf_hub_dataset(self.spec, hub_handle, trace=trace)

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


def openai_mrcr_loader_hf_hub_dataset(spec: DatasetSpec, hub_handle: Optional[DatasetHubHandle] = None, *, trace=None) -> DataSource:
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
    needle_type = loader_params.get('needle_type')
    max_content_window = loader_params.get('max_content_window')
    tokenizer_type = loader_params.get('tokenizer_type')
    tokenizer_name_or_path = loader_params.get("tokenizer_name_or_path")
    #data_files = loader_params.get("data_files")
    #local_dir = loader_params.get("local_dir")
    #subset_for_metadata = subset
    streaming = False
    cache_path = None

    logger.info(
        "Loading HF dataset hub_id={} subset={} split={} streaming={} cache={}",
        hub_id,
        "default",
        split,
        streaming_override if streaming_override is not None else False,
        "auto",
    )
    filename_list = get_needle_filename_list(needle_type)
    df_benchmark = _load_data(filename_list)
    benchmark = Dataset.from_pandas(df_benchmark)
    func = partial(global_filter, max_content_window=max_content_window,
                   tokenizer_type = tokenizer_type,
                    tokenizer_name_or_path = tokenizer_name_or_path)
    benchmark = benchmark.filter(func)
    benchmark = benchmark.to_list()
    
    limit = loader_params.get("limit")
    if limit is not None:
        benchmark = _apply_limit(benchmark, limit)
    #print("benchmark:", benchmark)    
    records = _maybe_apply_preprocess(
            benchmark,
            spec,
            data_path=hub_id,
            trace=trace,
    )
    #print("records:", records)
    metadata_payload = {
        "loader": "hf_hub",
        "hub_id": hub_id,
        "source": source,
        "split": split,
        "subset": "default",
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