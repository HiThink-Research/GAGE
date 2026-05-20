"""Class-based MMMU multimodal preprocessor."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.utils.multimodal import merge_multimodal_inputs
from gage_eval.assets.datasets.utils.rendering import set_render_flags

_MMMU_IMAGE_FIELD_COUNT = 7  # MMMU records expose numbered image_1..image_7 fields.


class MMMUMultimodalPreprocessor(BasePreprocessor):
    """Ensure MMMU samples have multi_modal_data/image filled from messages."""

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sample = dict(record)
        _attach_image_fragments(sample)
        merge_multimodal_inputs(sample)
        set_render_flags(
            sample,
            mode="preprocess",
            source="manual",
            rendered_by="preprocess",
            cache_suffix="-converted",
            overwrite=False,
        )
        sample.setdefault("inputs", {})
        return sample


def _attach_image_fragments(
    sample: Dict[str, Any],
) -> None:
    image_urls = _image_urls(sample)
    if not image_urls or _messages_have_image_fragments(sample.get("messages")):
        return
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        return
    target = _first_user_message(messages)
    if target is None:
        return
    content = target.get("content")
    target["content"] = [
        {"type": "image_url", "image_url": {"url": url}}
        for url in image_urls
    ] + _content_to_fragments(content)


def _image_urls(sample: Dict[str, Any]) -> list[str]:
    urls: list[str] = []

    def add(value: Any) -> None:
        url = _image_url(value)
        if url and url not in urls:
            urls.append(url)

    images = sample.get("images")
    if isinstance(images, list):
        for image in images:
            add(image)
    else:
        add(images)
    for index in range(1, _MMMU_IMAGE_FIELD_COUNT + 1):
        add(sample.get(f"image_{index}"))
    return urls


def _image_url(value: Any) -> str | None:
    if isinstance(value, str):
        return value if _is_supported_image_ref(value) else None
    if isinstance(value, dict):
        for key in ("path", "url", "image_url"):
            nested = value.get(key)
            if isinstance(nested, str):
                return nested if _is_supported_image_ref(nested) else None
            if isinstance(nested, dict) and isinstance(nested.get("url"), str):
                url = nested["url"]
                return url if _is_supported_image_ref(url) else None
    return None


def _is_supported_image_ref(value: str) -> bool:
    return value.startswith(("http://", "https://", "data:", "file:"))


def _messages_have_image_fragments(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        fragments = content if isinstance(content, list) else [content]
        for fragment in fragments:
            if isinstance(fragment, dict) and fragment.get("type") == "image_url":
                return True
    return False


def _first_user_message(messages: list[Any]) -> dict[str, Any] | None:
    for message in messages:
        if isinstance(message, dict) and str(message.get("role", "")).lower() == "user":
            return message
    return None


def _content_to_fragments(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in content if item is not None]
    if isinstance(content, dict):
        return [content]
    if content is None:
        return []
    return [{"type": "text", "text": str(content)}]


__all__ = ["MMMUMultimodalPreprocessor"]
