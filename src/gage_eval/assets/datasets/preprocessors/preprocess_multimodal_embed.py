"""Embed local multimodal references into data URLs and surface multi_modal_data upfront."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.assets.datasets.converters.image_utils import (
    embed_local_image_as_data_url,
    embed_local_message_images,
)


def convert_sample_to_inputs(
    sample: Dict[str, Any],
    *,
    message_content_field: str = "messages.0.content",
    image_field: Optional[str] = "messages.0.content.0.image_url.url",
    cache_dir: Optional[str] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Normalize多模态输入，将本地图片路径转换为 data URL，并回填到 multi_modal_data。

    返回值遵循 DatasetPreprocessor 协议，供 loader_utils.apply_preprocess 写入 sample["inputs"]。
    """

    # 1) 尝试在 messages.content 中批量转换本地 image_url
    embed_local_message_images(
        sample,
        content_field=message_content_field,
        cache_dir=cache_dir,
        strict=strict,
    )

    # 2) 兼容单路径字段（例如 legacy 数据）转换
    if image_field:
        embed_local_image_as_data_url(sample, image_field=image_field, cache_dir=cache_dir, strict=strict)

    # 3) 收集 data URL 或现有的 multi_modal_data.image 列表
    inputs = dict(sample.get("inputs") or {})
    mm = dict(inputs.get("multi_modal_data") or {})

    # 从 messages 提取到的 image_url 已经替换为 data URL，可直接放入 multi_modal_data
    message_images = _collect_image_urls(sample, content_field=message_content_field)
    if message_images:
        mm["image"] = message_images

    # 如果已有 multi_modal_data.image 且非空，保留并与 message_images 合并去重
    existing_images = mm.get("image") or mm.get("images")
    if existing_images:
        merged = []
        for src in existing_images:
            if src not in merged:
                merged.append(src)
        for src in message_images:
            if src not in merged:
                merged.append(src)
        mm["image"] = merged

    if mm:
        inputs["multi_modal_data"] = mm
    return inputs


def _collect_image_urls(sample: Dict[str, Any], *, content_field: str) -> list[str]:
    """从 messages.content 中收集已标准化的 image_url.url 字符串。"""

    # 按层级解析嵌套字段，兼容列表/字典
    parts = content_field.split(".")
    current: Any = sample
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                idx = int(part)
            except ValueError:
                return []
            if 0 <= idx < len(current):
                current = current[idx]
            else:
                return []
        else:
            return []
        if current is None:
            return []

    urls: list[str] = []
    if isinstance(current, list):
        for entry in current:
            if not isinstance(entry, dict):
                continue
            if entry.get("type") != "image_url":
                continue
            image_url = entry.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
            else:
                url = image_url
            if isinstance(url, str):
                urls.append(url)
    return urls
