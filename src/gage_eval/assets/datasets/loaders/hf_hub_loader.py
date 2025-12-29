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

from gage_eval.assets.datasets.loaders.loader_utils import (
    apply_default_params,
    apply_bundle,
    apply_preprocess,
    build_bundle_context,
    build_preprocess_context,
    inject_default_params,
    resolve_doc_to_callable,
    )
from gage_eval.observability.config import get_observability_config
from gage_eval.registry import registry
import logging
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError # Added import

logger = logging.getLogger(__name__)


@registry.asset(
    "dataset_loaders",
    "hf_hub",
    desc="Remote dataset loader for HuggingFace Hub / ModelScope",
    tags=("remote", "huggingface"),
    supports_streaming=True,
)
class HuggingFaceDatasetLoader(DatasetLoader):
    def load(self, hub_handle: Optional[DatasetHubHandle], *, trace=None) -> DataSource:
        return load_hf_hub_dataset(self.spec, hub_handle, trace=trace)


registry.register(
    "dataset_loaders",
    "modelscope",
    HuggingFaceDatasetLoader,
    desc="Remote dataset loader for ModelScope",
    tags=("remote", "modelscope"),
    supports_streaming=False,
)

registry.register(
    "dataset_loaders",
    "ms_hub",
    HuggingFaceDatasetLoader,
    desc="Remote dataset loader for ModelScope (ms_hub alias)",
    tags=("remote", "modelscope"),
    supports_streaming=False,
)


def load_hf_hub_dataset(spec: DatasetSpec, hub_handle: Optional[DatasetHubHandle] = None, *, trace=None) -> DataSource:
    """Download/cache a remote dataset and expose it as a DataSource."""

    try:
        import datasets  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "`datasets` package is required to load HuggingFace/ModelScope datasets"
        ) from exc

    _ensure_hf_list_feature_alias()

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

    subset_for_metadata = subset
    streaming = streaming_override if streaming_override is not None else _should_stream_hf(loader_params, source)
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

    doc_to_text = resolve_doc_to_callable(spec, "doc_to_text")
    doc_to_visual = resolve_doc_to_callable(spec, "doc_to_visual")
    doc_to_audio = resolve_doc_to_callable(spec, "doc_to_audio")

    if streaming:
        dataset = _download_dataset(
            datasets_module=datasets,
            hub_id=hub_id,
            source=source,
            split=split,
            subset=subset_for_metadata,
            revision=revision,
            trust_remote=trust_remote,
            load_kwargs=load_kwargs,
            streaming=True,
            builder_name=builder_name,
            data_files=data_files,
        )
        records: Iterable[Dict[str, Any]] = dataset
        logger.info("Streaming dataset ready ({})", hub_id)
        limit = loader_params.get("limit")
        if limit is not None:
            records = _limit_iterable(records, limit)
        records = apply_bundle(
            records,
            spec,
            data_path=hub_id,
            doc_to_text=doc_to_text,
            doc_to_visual=doc_to_visual,
            doc_to_audio=doc_to_audio,
            trace=trace,            
        )
        records = apply_preprocess(
            records,
            spec,
            data_path=hub_id,
            doc_to_text=doc_to_text,
            doc_to_visual=doc_to_visual,
            doc_to_audio=doc_to_audio,
            trace=trace,
        )
        records = apply_default_params(records, spec)
    else:
        dataset, cache_path = _load_or_cache_dataset(
            datasets_module=datasets,
            hub_id=hub_id,
            source=source,
            split=split,
            subset=subset_for_metadata,
            revision=revision,
            trust_remote=trust_remote,
            load_kwargs=load_kwargs,
            cache_dir=loader_params.get("cache_dir"),
            builder_name=builder_name,
            data_files=data_files,
        )

        if loader_params.get("shuffle"):
            dataset = dataset.shuffle(seed=loader_params.get("seed"))

        limit = loader_params.get("limit")
        if limit is not None:
            dataset = _apply_limit(dataset, limit)

        dataset = _maybe_apply_bundle(
            dataset,
            spec,
            data_path=hub_id,
            doc_to_text=doc_to_text,
            doc_to_visual=doc_to_visual,
            doc_to_audio=doc_to_audio,
            trace=trace,
        )

        dataset = _maybe_apply_preprocess(
            dataset,
            spec,
            data_path=hub_id,
            doc_to_text=doc_to_text,
            doc_to_visual=doc_to_visual,
            doc_to_audio=doc_to_audio,
            trace=trace,
        )
        dataset = _inject_default_params(dataset, spec)
        records = dataset
        logger.info("Dataset materialized at {}", cache_path)

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


def _ensure_hf_list_feature_alias() -> None:
    """Adds a compatibility alias for legacy HF datasets (`_type: List`)."""

    try:
        from datasets.features.features import _FEATURE_TYPES, Sequence  # type: ignore
    except Exception:
        return

    if "List" not in _FEATURE_TYPES:
        _FEATURE_TYPES["List"] = Sequence


def _load_or_cache_dataset(
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

def _maybe_apply_bundle(
    dataset,
    spec: DatasetSpec,
    *,
    data_path: str,
    doc_to_text=None,
    doc_to_visual=None,
    doc_to_audio=None,
    trace=None,
    observability_config=None,
):
    ctx = build_bundle_context(spec, data_path=data_path)
    if not ctx:
        return dataset
    config = observability_config or get_observability_config()

    def _map_fn(sample: Dict[str, Any]):
        new_sample = dict(sample)
        new_sample.setdefault("_dataset_id", spec.dataset_id)
        new_sample.setdefault("_dataset_metadata", {"path": data_path})
        new_sample = ctx.handle.apply(
            new_sample,
            dataset_id=spec.dataset_id,
            dataset_metadata=new_sample.get("_dataset_metadata"),
            trace=trace,
            observability_config=config,
            **ctx.kwargs,
        )
        return new_sample

    return dataset.map(_map_fn)

def _maybe_apply_preprocess(
    dataset,
    spec: DatasetSpec,
    *,
    data_path: str,
    doc_to_text=None,
    doc_to_visual=None,
    doc_to_audio=None,
    trace=None,
    observability_config=None,
):
    ctx = build_preprocess_context(spec, data_path=data_path)
    if not ctx:
        return dataset
    config = observability_config or get_observability_config()

    def _map_fn(sample: Dict[str, Any]):
        new_sample = dict(sample)
        new_sample.setdefault("_dataset_id", spec.dataset_id)
        new_sample.setdefault("_dataset_metadata", {"path": data_path})
        new_sample = ctx.handle.apply(
            new_sample,
            dataset_id=spec.dataset_id,
            dataset_metadata=new_sample.get("_dataset_metadata"),
            doc_to_text=doc_to_text,
            doc_to_visual=doc_to_visual,
            doc_to_audio=doc_to_audio,
            trace=trace,
            observability_config=config,
            **ctx.kwargs,
        )
        return new_sample
    return [_map_fn(item) for item in dataset]
    #return dataset.map(_map_fn)


def _inject_default_params(dataset, spec: DatasetSpec):
    params = spec.params.get("default_params")
    if not params:
        return dataset

    def _map_fn(sample: Dict[str, Any]):
        return inject_default_params(dict(sample), params)

    return dataset.map(_map_fn)


def _apply_limit(dataset, limit):
    if isinstance(limit, float):
        if not 0 < limit <= 1:
            raise ValueError("Float limit must be between 0 and 1")
        size = int(len(dataset) * limit)
        return dataset.select(range(size))
    if isinstance(limit, int):
        if limit < 0:
            raise ValueError("Limit must be non-negative")
    return dataset.select(range(min(limit, len(dataset))))


def _limit_iterable(records: Iterable[Dict[str, Any]], limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    if limit is None:
        return records

    def generator():
        for idx, record in enumerate(records):
            if limit is not None and idx >= limit:
                break
            yield record

    return generator()


def _should_stream_hf(args: Dict[str, Any], source: str) -> bool:
    flag = args.get("streaming")
    if flag is not None:
        return bool(flag)
    if source == "modelscope":
        return False
    return bool(args.get("hf_streaming"))


def _normalize_source(raw: str) -> str:
    normalized = raw.lower()
    if normalized not in {"huggingface", "modelscope", "local"}:
        raise ValueError(f"Unsupported dataset source: {raw}")
    return normalized


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
