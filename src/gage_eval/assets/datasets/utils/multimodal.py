"""Helpers for transforming image references inside dataset records."""

from __future__ import annotations

import base64
import mimetypes
import hashlib
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from io import BytesIO

from gage_eval.assets.datasets.utils.normalization import ensure_chat_template_flags
from gage_eval.assets.datasets.utils.mapping import extract_field


def collect_content_fragments(sample: Dict[str, Any], *, content_field: str, content_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Collect content fragments for multimodal fields, optionally resolving relative paths."""

    fragments = []
    content = extract_field(sample, content_field)
    if isinstance(content, list):
        fragments = [frag for frag in content if frag is not None]
    elif content is not None:
        fragments = [content]

    if not content_root:
        return fragments
    resolved_root = _expand_env(content_root)
    resolved: List[Dict[str, Any]] = []
    for frag in fragments:
        if isinstance(frag, dict) and frag.get("type") in {"image_url", "audio_url", "video_url", "file_url"}:
            key = frag["type"]
            payload = frag.get(key) or {}
            if isinstance(payload, dict) and isinstance(payload.get("url"), str) and not payload["url"].startswith(("http://", "https://", "data:")):
                payload = dict(payload)
                payload["url"] = _join_root(resolved_root, payload["url"])
                frag = dict(frag)
                frag[key] = payload
        resolved.append(frag)
    return resolved

# ---------------------------------------------------------------------------
# 轻量媒体处理与嵌入
# ---------------------------------------------------------------------------


def embed_local_image_as_data_url(
    sample: Dict[str, Any],
    *,
    image_field: str = "messages.0.content.0.image_url.url",
    strict: bool = False,
    cache_dir: Optional[str] = None,
    content_root: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Convert本地 image_path 为 data URL，避免远端 OpenAI 接口无法访问本地文件。"""

    path_value = _extract_nested(sample, image_field)
    if not path_value:
        return None
    if not isinstance(path_value, str):
        return None
    if path_value.startswith(("http://", "https://", "data:")):
        return {"type": "image_url", "image_url": {"url": path_value}}

    root_dir = content_root or _resolve_content_root(sample)
    root_path = Path(root_dir).expanduser() if root_dir else None
    resolved_path = Path(path_value)
    if not resolved_path.is_absolute():
        # 1) 尝试按当前工作目录直接解析
        try_candidates = []
        try:
            try_candidates.append(resolved_path.expanduser().resolve())
        except Exception:
            try_candidates.append(resolved_path.expanduser())

        # 2) 若路径前缀已包含 root 父目录名，避免二次拼接 root
        if root_path:
            parts = list(resolved_path.parts)
            root_parent = root_path.parent
            if root_parent and parts and parts[0] == root_parent.name:
                candidate_rel = Path(*parts[1:])
                try_candidates.append(root_parent.joinpath(candidate_rel).expanduser().resolve())

        # 3) 常规 root 拼接
        if root_path:
            try_candidates.append(root_path.joinpath(resolved_path).expanduser().resolve())

        # 4) 兜底：直接 expanduser
        try_candidates.append(resolved_path.expanduser())

        resolved_path = next((c for c in try_candidates if isinstance(c, Path) and c.exists()), try_candidates[-1])

    if not isinstance(resolved_path, Path) or not resolved_path.exists():
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


def encode_pil_to_data_url(pil_image, format: str = "PNG", **save_kwargs: Any) -> str:
    """将 PIL Image 编码为 data URL，供远端 HTTP 接口消费。"""

    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("encode_pil_to_data_url requires Pillow installed") from exc

    if not isinstance(pil_image, Image.Image):
        raise TypeError("encode_pil_to_data_url expects a PIL.Image instance")

    buf = BytesIO()
    pil_image.save(buf, format=format, **save_kwargs)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime_type = mimetypes.types_map.get(f".{format.lower()}", f"image/{format.lower()}")
    return f"data:{mime_type};base64,{encoded}"


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


def _resolve_content_root(sample: Dict[str, Any]) -> Optional[Path]:
    metadata = sample.get("metadata") or {}
    root = metadata.get("content_root") or metadata.get("image_root")
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
    content_root: Optional[str] = None,
) -> Optional[list[Dict[str, Any]]]:
    """Convert所有消息中的本地 image_url 路径。

    MMMU 等任务的题干往往在同一条消息中穿插多张图片。该辅助函数
    逐个检测 content 列表中的 ``image_url`` 项并复用
    :func:`embed_local_image_as_data_url` 将本地路径转换为 data URL，
    使远端推理后端无需直接访问文件系统。
    """

    root = _derive_root(sample, content_root)
    if root:
        try:
            resolved = Path(root).expanduser().resolve()
            meta = sample.get("metadata") or {}
            meta["content_root"] = str(resolved)
            # 兼容性保留
            if not meta.get("image_root"):
                meta["image_root"] = str(resolved)
            sample["metadata"] = meta
            content_root = str(resolved)
        except Exception:
            pass

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
        converted = embed_local_image_as_data_url(
            sample,
            image_field=field,
            strict=strict,
            cache_dir=cache_dir,
            content_root=content_root,
        )
        if converted:
            attachments.append(converted)

    return attachments or None


def _derive_root(sample: Dict[str, Any], explicit_root: Optional[str]) -> Optional[str]:
    """Infer a filesystem root for local media.

    优先级：
    1) 显式传入的 content_root
    2) sample.metadata.content_root (or image_root for compat)
    3) 数据集路径的父目录（_dataset_metadata.path）
    """

    if explicit_root:
        return os.path.expandvars(explicit_root)
    meta = sample.get("metadata") or {}
    root = meta.get("content_root") or meta.get("image_root")
    if root:
        return root
    ds_meta = sample.get("_dataset_metadata") or {}
    ds_path = ds_meta.get("path")
    if isinstance(ds_path, str):
        try:
            return str(Path(ds_path).expanduser().resolve().parent)
        except Exception:
            return str(Path(ds_path).expanduser().parent)
    return None


def resolve_media_path(path_value: Any, root: Optional[str] = None) -> Optional[str]:
    """Resolve media path with optional root; keep URL/data URL unchanged."""

    if path_value in (None, ""):
        return None
    path = str(path_value)
    if path.startswith(("http://", "https://", "file://", "data:")):
        return path
    if root:
        return str(Path(root).joinpath(path).as_posix())
    return path


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


# ---------------------------------------------------------------------------
# 多模态收集与合并（从 manager 迁移）
# ---------------------------------------------------------------------------

def merge_multimodal_inputs(sample: Dict[str, Any]) -> None:
    """归并 multi_modal_data，并保持与 messages 同步。

    - 兼容 legacy inputs（str/list），统一包装为 dict。
    - 合并 messages/doc_to_* 中的媒体引用，写入 inputs.multi_modal_data。
    - 同步 _media_meta.images，去重并可选嵌入本地媒体。
    """

    raw_inputs = sample.get("inputs")
    inputs: Dict[str, Any]
    if isinstance(raw_inputs, dict):
        inputs = dict(raw_inputs)
    elif isinstance(raw_inputs, list):
        # 兼容 preprocess 返回的 messages 列表
        if not sample.get("messages"):
            sample["messages"] = raw_inputs
        inputs = {}
    elif isinstance(raw_inputs, str):
        sample.setdefault("prompt", raw_inputs)
        inputs = {"prompt": raw_inputs}
    else:
        inputs = {}
    if sample.get("prompt") and "prompt" not in inputs:
        inputs["prompt"] = sample["prompt"]
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

    _sync_multimodal_with_messages(sample)
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


def _sync_multimodal_with_messages(sample: Dict[str, Any]) -> None:
    """仅保留 messages 中引用的媒体，清理未引用资源。"""

    inputs = sample.get("inputs")
    if not isinstance(inputs, dict):
        return
    mm = inputs.get("multi_modal_data")
    if not isinstance(mm, dict):
        return

    def _collect_from_messages(target_type: str) -> list[str]:
        urls: list[str] = []
        messages = sample.get("messages")
        if not isinstance(messages, list):
            return urls
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for fragment in content:
                if not isinstance(fragment, dict):
                    continue
                if fragment.get("type") != target_type:
                    continue
                url = fragment.get(target_type)
                if isinstance(url, dict):
                    url = url.get("url")
                elif url is None:
                    url = fragment.get("url")
                if isinstance(url, str):
                    urls.append(url)
        return urls

    keep_images = set(_collect_from_messages("image_url"))
    keep_audios = set(_collect_from_messages("audio_url"))
    keep_videos = set(_collect_from_messages("video_url"))
    keep_files = set(_collect_from_messages("file_url"))

    def _filter(kind: str, allowed: set[str]) -> None:
        if not allowed:
            return
        values = mm.get(kind) or mm.get(f"{kind}s")
        if values is None:
            return
        items = values if isinstance(values, list) else [values]
        filtered = [v for v in items if isinstance(v, str) and v in allowed]
        mm.pop(f"{kind}s", None)
        if filtered:
            mm[kind] = filtered
        else:
            mm.pop(kind, None)

    _filter("image", keep_images)
    _filter("audio", keep_audios)
    _filter("video", keep_videos)
    _filter("file", keep_files)

    inputs["multi_modal_data"] = mm
    sample["inputs"] = inputs

    media_meta = sample.get("_media_meta")
    if isinstance(media_meta, dict) and "images" in media_meta and keep_images:
        imgs = media_meta.get("images")
        if isinstance(imgs, list):
            media_meta["images"] = [img for img in imgs if isinstance(img, dict) and img.get("url") in keep_images]
        sample["_media_meta"] = media_meta


def _maybe_embed_local_media(sample: Dict[str, Any]) -> None:
    """按需将本地媒体转 data URL。"""

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
    ensure_chat_template_flags(sample)


def _embed_path_to_data_url(path: str) -> str:
    """读取本地文件并编码为 data URL，带 LRU 缓存避免重复 IO/编码。"""

    resolved = Path(path)
    mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    content = resolved.read_bytes()
    encoded = _encode_base64(content, use_process_pool=_should_use_process_pool(resolved, content))
    return f"data:{mime};base64,{encoded}"


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


def _join_root(root: str, path: str) -> str:
    if not root:
        return path
    if path.startswith("/"):
        return path
    if root.endswith("/"):
        return root + path
    return f"{root.rstrip('/')}/{path}"


def _expand_env(value: str) -> str:
    """Expand ${VAR} and ${VAR:-default} patterns with env fallback."""

    if value is None:
        return value
    pattern = re.compile(r"\$\{([^}:]+):-(.+)\}")
    match = pattern.search(value)
    if match:
        var, default = match.groups()
        if os.environ.get(var):
            return os.path.expandvars(pattern.sub(f"${{{var}}}", value))
        return pattern.sub(default, value)
    return os.path.expandvars(os.path.expanduser(value))


def _env_int(key: str) -> Optional[int]:
    """从环境变量读取 int，无法解析时返回 None。"""

    raw = os.environ.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


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
