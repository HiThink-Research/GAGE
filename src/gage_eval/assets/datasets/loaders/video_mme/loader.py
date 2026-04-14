"""Video-MME dataset loader that filters samples by local video availability."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

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
import logging

logger = logging.getLogger(__name__)


def _scan_video_dir(video_dir: str) -> Set[str]:
    """Scan directory for {video_id}.mp4 files and return available video IDs."""
    path = Path(video_dir).expanduser().resolve()
    if not path.exists():
        logger.warning("Video directory does not exist: {}", path)
        return set()
    return {entry.stem for entry in path.iterdir() if entry.is_file() and entry.suffix.lower() == ".mp4"}


def _inject_local_video(
    records: Iterable[Dict[str, Any]],
    video_dir: Path,
    available_ids: Set[str],
) -> Iterable[Dict[str, Any]]:
    """Inject local_video_path and filter out samples without a local video."""
    dropped = 0
    for record in records:
        sample = dict(record)
        video_id = sample.get("videoID")
        if not video_id:
            dropped += 1
            continue
        video_id_str = str(video_id).strip()
        if video_id_str not in available_ids:
            dropped += 1
            continue
        sample["local_video_path"] = str(video_dir / f"{video_id_str}.mp4")
        yield sample
    if dropped:
        logger.info("Dropped %d samples without local video", dropped)


@registry.asset(
    "dataset_loaders",
    "video_mme_hf_hub",
    desc="Video-MME loader with local video filtering",
    tags=("remote", "huggingface", "video_mme"),
    supports_streaming=True,
)
class VideoMMEHFDatasetLoader(DatasetLoader):
    def load(self, hub_handle: Optional[DatasetHubHandle], *, trace=None) -> DataSource:
        return load_video_mme_hf_dataset(self.spec, hub_handle, trace=trace)


def load_video_mme_hf_dataset(
    spec: DatasetSpec,
    hub_handle: Optional[DatasetHubHandle] = None,
    *,
    trace=None,
) -> DataSource:
    try:
        import datasets  # type: ignore
    except ImportError as exc:
        raise RuntimeError("`datasets` package is required") from exc

    from gage_eval.assets.datasets.loaders.hf_hub_loader import (
        _download_dataset,
        _load_or_cache_dataset,
        _should_stream_hf,
        _normalize_source,
    )

    metadata = dict(hub_handle.metadata) if hub_handle and hub_handle.metadata else {}
    loader_params = dict(metadata)
    loader_params.update(spec.params)
    streaming_override = None if hub_handle is None else hub_handle.streaming

    hub_id = loader_params.get("hub_id") or loader_params.get("path")
    if not hub_id:
        raise ValueError(f"Dataset '{spec.dataset_id}' missing 'hub_id' argument")

    video_dir = loader_params.get("video_dir", "/mnt/aime_data_ssd/user_workspace/zhuwenqiao/data/video-mme")
    available_ids = _scan_video_dir(video_dir)
    logger.info("VideoMME loader found {} local videos in {}", len(available_ids), video_dir)

    default_source = "modelscope" if spec.loader == "modelscope" else "huggingface"
    source = _normalize_source(loader_params.get("source", default_source))
    split = loader_params.get("split", "train")
    subset = loader_params.get("subset")
    revision = loader_params.get("revision")
    trust_remote = loader_params.get("trust_remote_code", True)
    load_kwargs = dict(loader_params.get("load_kwargs", {}))
    builder_name = loader_params.get("builder_name")
    data_files = loader_params.get("data_files")

    streaming = streaming_override if streaming_override is not None else _should_stream_hf(loader_params, source)
    if data_files:
        streaming = False
    cache_path = None

    doc_to_text = resolve_doc_to_callable(spec, "doc_to_text")
    doc_to_visual = resolve_doc_to_callable(spec, "doc_to_visual")
    doc_to_audio = resolve_doc_to_callable(spec, "doc_to_audio")

    if streaming:
        dataset = _download_dataset(
            datasets_module=datasets,
            hub_id=hub_id,
            source=source,
            split=split,
            subset=subset,
            revision=revision,
            trust_remote=trust_remote,
            load_kwargs=load_kwargs,
            streaming=True,
            builder_name=builder_name,
            data_files=data_files,
        )
        records: Iterable[Dict[str, Any]] = dataset
        limit = loader_params.get("limit")
        if limit is not None:
            records = _limit_iterable(records, limit)
    else:
        dataset, cache_path = _load_or_cache_dataset(
            datasets_module=datasets,
            hub_id=hub_id,
            source=source,
            split=split,
            subset=subset,
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
        records = dataset

    # Inject local video paths and filter missing samples.
    records = _inject_local_video(records, Path(video_dir).expanduser().resolve(), available_ids)

    # Apply preprocessor.
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

    metadata_payload = {
        "loader": "video_mme_hf_hub",
        "hub_id": hub_id,
        "source": source,
        "split": split,
        "subset": subset or "default",
        "revision": revision or "latest",
        "cache_path": cache_path,
        "streaming": streaming,
        "video_dir": video_dir,
        "local_videos": len(available_ids),
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


def _limit_iterable(records: Iterable[Dict[str, Any]], limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    if limit is None:
        return records

    def generator():
        for idx, record in enumerate(records):
            if limit is not None and idx >= limit:
                break
            yield record

    return generator()


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
