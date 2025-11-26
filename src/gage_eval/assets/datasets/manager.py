"""Data management utilities shared by all pipelines."""

from __future__ import annotations

import base64
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, Optional

from loguru import logger

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.assets.datasets.validation import SampleValidator, build_validator


@dataclass
class DataSource:
    """Represents a reusable dataset along with optional doc_to hooks."""

    dataset_id: str
    records: Iterable[Dict[str, Any]]
    doc_to_text: Optional[Callable[[Dict[str, Any]], str]] = None
    doc_to_visual: Optional[Callable[[Dict[str, Any]], Any]] = None
    doc_to_audio: Optional[Callable[[Dict[str, Any]], Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    validator: Optional[SampleValidator] = field(default=None, repr=False)
    streaming: bool = False


class DataManager:
    """Central entry point for dataset loading, caching and schema mapping.

    The implementation borrows ideas from LMMS' ``doc_to_*`` adapter
    pattern as well as llm-eval's ``data_server.py`` queue-based design.
    For now the class keeps things simple: datasets are assumed to be
    in-memory iterables. Future iterations can mount streaming sources
    or integrate HuggingFace datasets by extending ``register_loader``.
    """

    def __init__(self) -> None:
        self._sources: Dict[str, DataSource] = {}

    def register_source(self, source: DataSource, trace: Optional[ObservabilityTrace] = None) -> None:
        if source.dataset_id in self._sources:
            raise ValueError(f"Dataset '{source.dataset_id}' is already registered")
        source.validator = build_validator(source.validation)
        self._sources[source.dataset_id] = source
        metadata = source.metadata or {}
        formatted_meta = ", ".join(f"{key}={value}" for key, value in metadata.items()) or "no-metadata"
        logger.info("Registered dataset '{}' ({})", source.dataset_id, formatted_meta)
        if trace:
            trace.emit(
                "dataset_registered",
                {
                    "dataset_id": source.dataset_id,
                    "metadata": metadata,
                },
            )

    def get(self, dataset_id: str) -> DataSource:
        try:
            return self._sources[dataset_id]
        except KeyError as exc:
            raise KeyError(f"Dataset '{dataset_id}' is not registered") from exc

    def iter_samples(self, dataset_id: str, trace: Optional[ObservabilityTrace] = None) -> Iterator[Dict[str, Any]]:
        """Yield normalized samples for the requested dataset.

        Args:
            dataset_id: Identifier passed to :meth:`register_source`.
            trace: Optional observability sink used to emit per-sample events.
        """

        logger.debug("Iterating samples for dataset_id='{}'", dataset_id)
        source = self.get(dataset_id)
        validator = source.validator
        adapters_context = {
            "doc_to_text": source.doc_to_text,
            "doc_to_visual": source.doc_to_visual,
            "doc_to_audio": source.doc_to_audio,
        }
        for index, record in enumerate(source.records):
            if not isinstance(record, dict):
                # 尝试解包单元素 list/tuple -> dict，避免 streaming 返回的包装结构
                if isinstance(record, (list, tuple)) and len(record) == 1 and isinstance(record[0], dict):
                    candidate = dict(record[0])
                else:
                    logger.warning(
                        "Skipping non-dict record dataset='%s' index=%s type=%s",
                        dataset_id,
                        index,
                        type(record).__name__,
                    )
                    continue
            else:
                candidate = dict(record)
            if validator:
                validated = validator.validate_raw(candidate, dataset_id=dataset_id, index=index, trace=trace)
                if validated is None:
                    continue
                candidate = validated
            normalized = dict(candidate)
            normalized.setdefault("_dataset_metadata", source.metadata or {})
            try:
                # Convert raw documents into the runtime-friendly schema without mutating the source record.
                has_inputs = "inputs" in normalized or "messages" in normalized
                if source.doc_to_text and not has_inputs:
                    normalized["text"] = source.doc_to_text(normalized)
                if source.doc_to_visual and not has_inputs:
                    normalized["visual"] = source.doc_to_visual(normalized)
                if source.doc_to_audio and not has_inputs:
                    normalized["audio"] = source.doc_to_audio(normalized)
                _merge_multimodal_inputs(normalized)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning(
                    "doc_to_* adapter failed for dataset='{}' index={} ({})",
                    dataset_id,
                    index,
                    exc,
                )
                if trace:
                    trace.emit(
                        "data_adapter_failed",
                        {
                            "dataset_id": dataset_id,
                            "index": index,
                            "error": str(exc),
                        },
                    )
                continue
            if trace:
                trace.emit(
                    "data_sample_emitted",
                    {
                        "dataset_id": dataset_id,
                        "index": index,
                        "keys": list(normalized.keys()),
                    },
                )
            logger.debug("Emitted normalized sample idx={} keys={}", index, list(normalized.keys()))
            yield normalized


def _merge_multimodal_inputs(sample: Dict[str, Any]) -> None:
    """Normalize multi_modal_data onto sample['inputs'] for downstream backends.

    - 优先使用 doc_to_visual / doc_to_audio 生成的字段。
    - 若未提供，则从 messages[].content[].image_url / audio_url 中兜底提取。
    - 不覆盖已有 multi_modal_data，仅做合并去重。
    """

    raw_inputs = sample.get("inputs")
    inputs: Dict[str, Any]
    if isinstance(raw_inputs, dict):
        inputs = dict(raw_inputs)
    elif isinstance(raw_inputs, list):
        # 兼容 preprocess 返回的 messages 列表：转存到 sample.messages，inputs 置空
        if not sample.get("messages"):
            sample["messages"] = raw_inputs
        inputs = {}
    else:
        inputs = {}
    sample["inputs"] = inputs
    mm = inputs.get("multi_modal_data") or {}

    def _dedup(seq):
        seen = set()
        out = []
        for item in seq:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    images, image_meta = _extract_images(sample)
    audios = _extract_audios(sample)
    videos = _extract_videos(sample)
    files = _extract_files(sample)

    if images:
        existing = mm.get("image") or mm.get("images") or []
        merged = list(existing) if isinstance(existing, list) else [existing]
        merged.extend(images)
        mm["image"] = _dedup([img for img in merged if img])
    if audios:
        existing = mm.get("audio") or mm.get("audios") or []
        merged = list(existing) if isinstance(existing, list) else [existing]
        merged.extend(audios)
        mm["audio"] = _dedup([a for a in merged if a])
    if videos:
        existing = mm.get("video") or mm.get("videos") or []
        merged = list(existing) if isinstance(existing, list) else [existing]
        merged.extend(videos)
        mm["video"] = _dedup([v for v in merged if v])
    if files:
        existing = mm.get("file") or mm.get("files") or []
        merged = list(existing) if isinstance(existing, list) else [existing]
        merged.extend(files)
        mm["file"] = _dedup([f for f in merged if f])

    if mm:
        inputs["multi_modal_data"] = mm
        sample["inputs"] = inputs
    if image_meta:
        media_meta = sample.get("_media_meta") or {}
        media_meta["images"] = image_meta
        sample["_media_meta"] = media_meta

    _maybe_embed_local_media(sample)


def _extract_images(sample: Dict[str, Any]) -> tuple[list[str], list[Dict[str, Any]]]:
    images: list[str] = []
    meta: list[Dict[str, Any]] = []

    def _from_visual(visual):
        if visual is None:
            return
        if isinstance(visual, str):
            images.append(visual)
            return
        if isinstance(visual, list):
            for item in visual:
                if isinstance(item, str):
                    images.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "image_url":
                        url = item.get("image_url")
                        detail = item.get("detail")
                        if isinstance(url, dict):
                            detail = detail or url.get("detail")
                            url = url.get("url")
                        if isinstance(url, str):
                            images.append(url)
                            meta.append({"url": url, "detail": detail} if detail else {"url": url})
                    elif "url" in item and isinstance(item.get("url"), str):
                        images.append(item["url"])
                        meta.append({"url": item["url"]})

    def _from_messages(messages):
        if not isinstance(messages, list):
            return
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for fragment in content:
                if not isinstance(fragment, dict):
                    continue
                if fragment.get("type") == "image_url":
                    url = fragment.get("image_url")
                    detail = fragment.get("detail")
                    if isinstance(url, dict):
                        detail = detail or url.get("detail")
                        url = url.get("url")
                    elif url is None:
                        url = fragment.get("url")
                    if isinstance(url, str):
                        images.append(url)
                        meta.append({"url": url, "detail": detail} if detail else {"url": url})

    _from_visual(sample.get("visual"))
    _from_messages(sample.get("messages"))
    return images, meta


def _extract_audios(sample: Dict[str, Any]) -> list[str]:
    audios: list[str] = []

    def _from_audio_field(audio):
        if audio is None:
            return
        if isinstance(audio, str):
            audios.append(audio)
            return
        if isinstance(audio, list):
            for item in audio:
                if isinstance(item, str):
                    audios.append(item)

    def _from_messages(messages):
        if not isinstance(messages, list):
            return
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for fragment in content:
                if not isinstance(fragment, dict):
                    continue
                if fragment.get("type") == "audio_url":
                    url = fragment.get("audio_url")
                    if isinstance(url, dict):
                        url = url.get("url")
                    elif url is None:
                        url = fragment.get("url")
                    if isinstance(url, str):
                        audios.append(url)

    _from_audio_field(sample.get("audio"))
    _from_messages(sample.get("messages"))
    return audios


def _extract_videos(sample: Dict[str, Any]) -> list[str]:
    videos: list[str] = []

    def _from_messages(messages):
        if not isinstance(messages, list):
            return
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for fragment in content:
                if not isinstance(fragment, dict):
                    continue
                if fragment.get("type") == "video_url":
                    url = fragment.get("video_url")
                    if isinstance(url, dict):
                        url = url.get("url")
                    elif url is None:
                        url = fragment.get("url")
                    if isinstance(url, str):
                        videos.append(url)

    _from_messages(sample.get("messages"))
    return videos


def _extract_files(sample: Dict[str, Any]) -> list[str]:
    files: list[str] = []

    def _from_messages(messages):
        if not isinstance(messages, list):
            return
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for fragment in content:
                if not isinstance(fragment, dict):
                    continue
                if fragment.get("type") == "file_url":
                    url = fragment.get("file_url")
                    if isinstance(url, dict):
                        url = url.get("url")
                    elif url is None:
                        url = fragment.get("url")
                    if isinstance(url, str):
                        files.append(url)

    _from_messages(sample.get("messages"))
    return files


def _maybe_embed_local_media(sample: Dict[str, Any]) -> None:
    """Optionally convert本地路径为 data URL，避免 HTTP 后端无法访问本地文件。

    触发条件：
    - 环境变量 `GAGE_EVAL_EMBED_LOCAL_MEDIA=1/true/on`，或
    - `_dataset_metadata.embed_local_media` 显式为真。
    """

    meta = sample.get("_dataset_metadata") or {}
    env_flag = os.environ.get("GAGE_EVAL_EMBED_LOCAL_MEDIA", "").lower() in {"1", "true", "yes", "on"}
    meta_flag = bool(meta.get("embed_local_media"))
    if not (env_flag or meta_flag):
        return

    base_path = meta.get("path")
    inputs = sample.get("inputs") or {}
    mm = inputs.get("multi_modal_data") or {}
    images = mm.get("image") or mm.get("images")
    audios = mm.get("audio") or mm.get("audios")
    videos = mm.get("video") or mm.get("videos")
    files = mm.get("file") or mm.get("files")

    max_workers = _env_int("GAGE_EVAL_EMBED_MEDIA_THREADS") or 0
    if images:
        mm["image"] = _embed_media_list(images, base_path, max_workers)
    if audios:
        mm["audio"] = _embed_media_list(audios, base_path, max_workers)
    if videos:
        mm["video"] = _embed_media_list(videos, base_path, max_workers)
    if files:
        mm["file"] = _embed_media_list(files, base_path, max_workers)

    inputs["multi_modal_data"] = mm
    sample["inputs"] = inputs


@lru_cache(maxsize=256)
def _embed_path_to_data_url(path: str) -> str:
    """读取本地文件并编码为 data URL，带 LRU 缓存避免重复 IO/编码。"""

    resolved = Path(path)
    mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    data = resolved.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _embed_media_list(media, base_path: Optional[str], max_workers: int) -> list[Any]:
    """将媒体引用列表转为 Data URL，支持可选线程池并保持顺序。"""

    items = media if isinstance(media, list) else [media]

    def _resolve_one(item):
        if not isinstance(item, str):
            return item
        if item.startswith(("http://", "https://", "data:")):
            return item
        try:
            resolved = Path(item)
            if not resolved.is_absolute() and base_path:
                resolved = Path(base_path).expanduser().resolve().parent.joinpath(item)
            resolved = resolved.expanduser().resolve()
            if not resolved.exists():
                return item
            return _embed_path_to_data_url(str(resolved))
        except Exception:
            return item

    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_resolve_one, items))
    return [_resolve_one(x) for x in items]
