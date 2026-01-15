"""Loader capable of pulling datasets from HuggingFace Hub or ModelScope."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.assets.datasets.hubs.base import DatasetHubHandle
from gage_eval.assets.datasets.loaders.base import DatasetLoader
from gage_eval.assets.datasets.manager import DataSource
import logging
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError # Added import

def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)


def _hash_config(hub_id: str, split: str, subset: Optional[str], revision: Optional[str], load_kwargs: Dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "hub_id": hub_id,
            "split": split,
            "subset": subset,
            "revision": revision,
            "load_kwargs": load_kwargs,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()

def load_or_cache_dataset(
    *,
    datasets_module,
    hub_id: str,
    source: str,
    split: str,
    subset: Optional[str],
    revision: Optional[str],
    trust_remote: bool,
    load_kwargs: Dict[str, Any],
    cache_dir: Optional[str],
    builder_name: Optional[str],
    data_files,
):
    cache_root = Path(cache_dir or os.getenv("GAGE_EVAL_DATA_CACHE") or Path(".cache/gage-eval/datasets")).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{_safe_name(hub_id)}-{_hash_config(hub_id, split, subset, revision, load_kwargs)}"

    if cache_path.exists():
        return datasets_module.load_from_disk(str(cache_path)), str(cache_path)

    dataset = _download_dataset(
        datasets_module=datasets_module,
        hub_id=hub_id,
        source=source,
        split=split,
        subset=subset,
        revision=revision,
        trust_remote=trust_remote,
        load_kwargs=load_kwargs,
        builder_name=builder_name,
        data_files=data_files,
    )

    materialized_path = str(cache_path)
    if source != "local":
        dataset.save_to_disk(materialized_path)
    else:
        materialized_path = hub_id
    return dataset, materialized_path

def _resolve_data_files_spec(spec, default_revision):
    try:  # pragma: no cover - optional dependency
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Specifying data_files requires the 'huggingface_hub' package") from exc

    def _resolve(entry):
        if isinstance(entry, dict):
            if "repo_id" in entry and "path" in entry:
                repo_revision = entry.get("revision") or default_revision or "main"
                repo_type = entry.get("repo_type", "dataset")
                filename = entry["path"]
                try:
                    return hf_hub_download(
                        repo_id=entry["repo_id"],
                        repo_type=repo_type,
                        filename=filename,
                        revision=repo_revision,
                    )
                except EntryNotFoundError:
                    base, ext = os.path.splitext(filename)
                    base_lower = base.lower()
                    fallback = None
                    if base_lower == "valid":
                        fallback = f"validation{ext}"
                    elif base_lower == "validation":
                        fallback = f"valid{ext}"
                    if fallback:
                        return hf_hub_download(
                            repo_id=entry["repo_id"],
                            repo_type=repo_type,
                            filename=fallback,
                            revision=repo_revision,
                        )
                    raise
            return {key: _resolve(value) for key, value in entry.items()}
        if isinstance(entry, list):
            return [_resolve(item) for item in entry]
        return entry

    return _resolve(spec)

def _download_dataset(
    *,
    datasets_module,
    hub_id: str,
    source: str,
    split: str,
    subset: Optional[str],
    revision: Optional[str],
    trust_remote: bool,
    load_kwargs: Dict[str, Any],
    streaming: bool = False,
    builder_name: Optional[str] = None,
    data_files=None,
):
    normalized_subset = None if subset in (None, "default") else subset

    if data_files:
        builder = builder_name or "json"
        resolved_files = _resolve_data_files_spec(data_files, revision)
        return datasets_module.load_dataset(
            path=builder,
            data_files=resolved_files,
            split=split,
            streaming=streaming,
            **load_kwargs,
        )
    if source == "modelscope":
        try:  # pragma: no cover - optional dependency
            from modelscope import MsDataset  # type: ignore
        except ImportError as exc:
            raise RuntimeError("ModelScope support requires the `modelscope` package") from exc

        dataset = MsDataset.load(
            dataset_name=hub_id,
            split=split,
            subset_name=normalized_subset,
            version=revision,
            trust_remote_code=trust_remote,
            **load_kwargs,
        )
        if not isinstance(dataset, datasets_module.Dataset):
            dataset = dataset.to_hf_dataset()
        return dataset

    try:
        return datasets_module.load_dataset(
            path=hub_id,
            name=normalized_subset,
            split=split,
            revision=revision,
            trust_remote_code=trust_remote,
            streaming=streaming,
            **load_kwargs,
        )
    except UnicodeDecodeError as exc:
        # Some hubs serve the dataset script gzipped, causing UTF-8 decode failures.
        if source == "huggingface":
            script_name = f"{hub_id.split('/')[-1]}.py"
            try:
                local_script = hf_hub_download(
                    repo_id=hub_id,
                    repo_type="dataset",
                    filename=script_name,
                    revision=revision,
                )
                return datasets_module.load_dataset(
                    path=local_script,
                    name=normalized_subset,
                    split=split,
                    revision=revision,
                    trust_remote_code=trust_remote,
                    streaming=streaming,
                    **load_kwargs,
                )
            except Exception:
                pass
        raise exc