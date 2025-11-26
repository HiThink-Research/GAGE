"""Helpers for transforming image references inside dataset records."""

from __future__ import annotations

import base64
import mimetypes
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional


def embed_local_image_as_data_url(
    sample: Dict[str, Any],
    *,
    image_field: str = "messages.0.content.0.image_url.url",
    strict: bool = False,
    cache_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Convert本地 image_path 为 data URL，避免远端 OpenAI 接口无法访问本地文件。"""

    path_value = _extract_nested(sample, image_field)
    if not path_value:
        return None
    if not isinstance(path_value, str):
        return None
    if path_value.startswith(("http://", "https://", "data:")):
        return {"type": "image_url", "image_url": {"url": path_value}}

    image_root = _resolve_image_root(sample)
    resolved_path = Path(path_value)
    if not resolved_path.is_absolute():
        if image_root:
            resolved_path = Path(image_root).joinpath(path_value)
        resolved_path = resolved_path.expanduser().resolve()

    if not resolved_path.exists():
        if strict:
            raise FileNotFoundError(f"Image file not found: {resolved_path}")
        return None

    mime_type = mimetypes.guess_type(resolved_path.name)[0] or "application/octet-stream"
    data_url = _cached_data_url(resolved_path, mime_type=mime_type, cache_dir=cache_dir)

    _assign_nested(sample, image_field, data_url)
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        metadata["image_url"] = data_url
    return {"type": "image_url", "image_url": {"url": data_url}}


def _extract_nested(sample: Dict[str, Any], field: Optional[str]) -> Any:
    if not field:
        return None
    current: Any = sample
    for part in field.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                return None
            if 0 <= index < len(current):
                current = current[index]
            else:
                return None
        else:
            return None
        if current is None:
            return None
    return current


def _assign_nested(sample: Dict[str, Any], field: str, value: Any) -> None:
    parts = field.split(".")
    current: Any = sample
    for idx, part in enumerate(parts):
        is_last = idx == len(parts) - 1
        if isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                raise ValueError(f"List index required for path segment '{part}'")
            while index >= len(current):
                current.append({})
            if is_last:
                current[index] = value
            else:
                if not isinstance(current[index], (dict, list)):
                    current[index] = {}
                current = current[index]
            continue
        if isinstance(current, dict):
            if is_last:
                current[part] = value
            else:
                if part not in current or not isinstance(current[part], (dict, list)):
                    current[part] = {}
                current = current[part]
            continue
        raise TypeError(f"Unsupported container type in assign path for segment '{part}'")


def _resolve_image_root(sample: Dict[str, Any]) -> Optional[Path]:
    metadata = sample.get("metadata") or {}
    root = metadata.get("image_root")
    if root:
        return Path(root).expanduser().resolve()
    dataset_meta = sample.get("_dataset_metadata") or {}
    dataset_path = dataset_meta.get("path")
    if dataset_path:
        return Path(dataset_path).expanduser().resolve().parent
    return None


def embed_local_message_images(
    sample: Dict[str, Any],
    *,
    content_field: str = "messages.0.content",
    strict: bool = False,
    cache_dir: Optional[str] = None,
) -> Optional[list[Dict[str, Any]]]:
    """Convert所有消息中的本地 image_url 路径。

    MMMU 等任务的题干往往在同一条消息中穿插多张图片。该辅助函数
    逐个检测 content 列表中的 ``image_url`` 项并复用
    :func:`embed_local_image_as_data_url` 将本地路径转换为 data URL，
    使远端推理后端无需直接访问文件系统。
    """

    content = _extract_nested(sample, content_field)
    if not isinstance(content, list):
        return None

    attachments: list[Dict[str, Any]] = []
    for idx, entry in enumerate(content):
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "image_url":
            continue
        field = f"{content_field}.{idx}.image_url.url"
        converted = embed_local_image_as_data_url(sample, image_field=field, strict=strict, cache_dir=cache_dir)
        if converted:
            attachments.append(converted)

    return attachments or None


def _cached_data_url(resolved_path: Path, *, mime_type: str, cache_dir: Optional[str]) -> str:
    """Encode文件并缓存 data URL，重复调用避免多次 base64 编码。"""

    cache_root = Path(cache_dir or os.environ.get("GAGE_VISUAL_CACHE_DIR", ".gage_cache/visual")).expanduser().resolve()
    content = resolved_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    cache_path = cache_root / f"{digest}.b64"
    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8")
        except Exception:
            pass

    encoded = _encode_base64(content, use_process_pool=_should_use_process_pool(resolved_path, content))
    data_url = f"data:{mime_type};base64,{encoded}"
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(data_url, encoding="utf-8")
    except Exception:
        # 缓存失败不影响主流程
        pass
    return data_url


_PROCESS_POOL: Optional[ProcessPoolExecutor] = None


def _encode_base64(content: bytes, *, use_process_pool: bool) -> str:
    if not use_process_pool:
        return base64.b64encode(content).decode("ascii")
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=1)
    future = _PROCESS_POOL.submit(_encode_bytes, content)
    return future.result()


def _encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _should_use_process_pool(resolved_path: Path, content: bytes) -> bool:
    flag = os.environ.get("GAGE_VISUAL_PROCESSPOOL")
    if flag is None:
        return False
    threshold_env = os.environ.get("GAGE_VISUAL_PROCESSPOOL_THRESHOLD")
    try:
        threshold = int(threshold_env) if threshold_env else 5 * 1024 * 1024
    except ValueError:
        threshold = 5 * 1024 * 1024
    size = len(content) if content else resolved_path.stat().st_size
    return size >= threshold
