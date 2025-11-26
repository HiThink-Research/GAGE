"""MMMU 多模态数据预处理器。

该预处理器从 messages 中抽取图片/音频路径，写入 sample['inputs']['multi_modal_data']，
以便本地 VLM（如 vlm_transformers）可以直接消费多模态数据。

注意：
- 这里仅整理路径，不做 Base64 编码或 PIL/Tensor 转换，重型处理仍交给 Backend。
- 本模块兼容实机 MMMU JSONL 格式（messages + image_url + text + choices + answer）。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


def _collect_media_paths(sample: Dict[str, Any], *, data_path: Optional[str]) -> Tuple[List[str], List[str]]:
    """从 messages 中抽取 image/audio 路径，支持本地路径及带 url 字段的 dict。"""

    messages = sample.get("messages")
    if not isinstance(messages, list):
        return [], []

    images: List[str] = []
    audios: List[str] = []

    def _to_path(value: Any) -> Optional[str]:
        if value is None:
            return None
        url = None
        if isinstance(value, dict):
            url = value.get("url")
        elif isinstance(value, str):
            url = value
        if not url:
            return None
        # 相对路径补齐到与 data_path 同目录，保持与其它预处理器一致的行为
        if data_path is not None and not os.path.isabs(url):
            url = os.path.join(os.path.dirname(data_path), url.lstrip("/"))
        return url

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for fragment in content:
            if not isinstance(fragment, dict):
                continue
            f_type = fragment.get("type")
            if f_type == "image_url":
                path = _to_path(fragment.get("image_url"))
                if path:
                    images.append(path)
            elif f_type == "image":
                path = _to_path(fragment.get("image"))
                if path:
                    images.append(path)
            elif f_type == "audio_url":
                path = _to_path(fragment.get("audio_url"))
                if path:
                    audios.append(path)

    return images, audios


def convert_sample_to_inputs(sample: Dict[str, Any], *, data_path: Optional[str] = None) -> Dict[str, Any]:
    """为 MMMU 样本构造 inputs.multi_modal_data。

    返回值会被 JSONL loader 写入 sample['inputs'] 字段：
      - image: List[str]，图片文件的本地路径或绝对 URL
      - audio: List[str]，音频文件路径（预留）
    """

    existing_inputs = sample.get("inputs")
    if isinstance(existing_inputs, dict) and isinstance(existing_inputs.get("multi_modal_data"), dict):
        # 已存在 multi_modal_data 时不重复覆盖，保持幂等
        return existing_inputs

    images, audios = _collect_media_paths(sample, data_path=data_path)
    mm: Dict[str, Any] = {}
    if images:
        mm["image"] = images
    if audios:
        mm["audio"] = audios

    inputs: Dict[str, Any] = existing_inputs if isinstance(existing_inputs, dict) else {}
    if mm:
        # 仅在检测到多模态数据时写入 multi_modal_data，避免干扰纯文本任务
        merged_mm = dict(inputs.get("multi_modal_data") or {})
        merged_mm.update(mm)
        inputs["multi_modal_data"] = merged_mm
    return inputs

