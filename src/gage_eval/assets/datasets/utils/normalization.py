"""Normalization helpers extracted from SampleStandardizer."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Union

try:  # Optional dependency; only used for isinstance checks
    from PIL import Image as _PILImage  # type: ignore
except ImportError:  # pragma: no cover - pillow is optional
    _PILImage = None


def ensure_sample_id(sample: Dict[str, Any], dataset_id: str) -> str:
    for key in ("id", "sample_id", "uid", "_id"):
        value = sample.get(key)
        if value not in (None, ""):
            return str(value)
    payload = json.dumps(sample, sort_keys=True, default=str)
    return hashlib.sha1(f"{dataset_id}:{payload}".encode("utf-8")).hexdigest()


def normalize_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = sample.get("messages")
    normalized: List[Dict[str, Any]] = []
    if isinstance(messages, list):
        normalized = [_normalize_message(msg) for msg in messages]
    else:
        prompt = sample.get("prompt") or sample.get("text") or sample.get("question")
        if prompt:
            normalized = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": str(prompt)}],
                }
            ]
    return _inject_modal_fragments(sample, normalized)


def normalize_choices(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    choices = sample.get("choices")
    if isinstance(choices, list) and choices:
        normalized: List[Dict[str, Any]] = []
        for idx, choice in enumerate(choices):
            normalized.append(_normalize_choice(choice, idx))
        return normalized

    fallback = None
    for key in ("answer", "reference_answer", "target", "expected", "label"):
        if sample.get(key) not in (None, ""):
            fallback = sample[key]
            break
    if fallback is None:
        return []
    return [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": str(fallback)}],
            },
        }
    ]


def normalize_sample(sample: Dict[str, Any], *, dataset_id: str, dataset_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalize core fields (id/messages/choices/metadata/etc.)."""

    sample["id"] = ensure_sample_id(sample, dataset_id)
    sample["_dataset_id"] = sample.get("_dataset_id") or dataset_id
    sample["_dataset_metadata"] = sample.get("_dataset_metadata") or (dataset_metadata or {})
    sample["messages"] = normalize_messages(sample)
    sample["choices"] = normalize_choices(sample)
    sample.setdefault("predict_result", sample.get("predict_result") or [])
    sample.setdefault("eval_result", sample.get("eval_result") or {})
    sample.setdefault("data_tag", sample.get("data_tag") or {})
    sample.setdefault("model_prompt_tmpl", sample.get("model_prompt_tmpl") or "")
    sample.setdefault("model_prompt_placeholder", sample.get("model_prompt_placeholder") or [])
    return sample


def ensure_chat_template_flags(sample: Dict[str, Any]) -> None:
    """Fill chat template flags when prompt exists but flags missing."""

    if "chat_template_mode" in sample or "prompt" not in sample:
        return
    sample["chat_template_mode"] = "preprocess"
    sample["rendered_by"] = "preprocess"
    sample["template_source"] = "manual"
    sample["cache_suffix"] = "-manual"


def get_prompt(sample: Dict[str, Any]) -> str | None:
    """Return prompt from sample or inputs."""

    if isinstance(sample.get("prompt"), str):
        return sample["prompt"]
    inputs = sample.get("inputs")
    if isinstance(inputs, dict) and isinstance(inputs.get("prompt"), str):
        return inputs["prompt"]
    return None


def list_images(sample: Dict[str, Any]) -> List[str]:
    """Collect image URLs from inputs.multi_modal_data and messages."""

    urls: List[str] = []
    inputs = sample.get("inputs")
    if isinstance(inputs, dict):
        mm = inputs.get("multi_modal_data") or {}
        if isinstance(mm, dict):
            if isinstance(mm.get("image"), list):
                urls.extend([u for u in mm["image"] if isinstance(u, str)])
            elif isinstance(mm.get("image"), str):
                urls.append(mm["image"])
    messages = sample.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for frag in content:
                if isinstance(frag, dict) and frag.get("type") == "image_url":
                    payload = frag.get("image_url")
                    if isinstance(payload, dict) and isinstance(payload.get("url"), str):
                        urls.append(payload["url"])
                    elif isinstance(payload, str):
                        urls.append(payload)
    seen = set()
    dedup: List[str] = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        dedup.append(u)
    return dedup


def _normalize_choice(choice: Any, default_index: int) -> Dict[str, Any]:
    if isinstance(choice, str):
        return {
            "index": default_index,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": choice}],
            },
        }
    if isinstance(choice, dict):
        message = choice.get("message")
        if isinstance(message, dict):
            normalized_message = _normalize_message(message, default_role="assistant")
        else:
            text_value = choice.get("text") or choice.get("answer") or ""
            normalized_message = {
                "role": "assistant",
                "content": [{"type": "text", "text": str(text_value)}],
            }
        return {
            "index": choice.get("index", default_index),
            "message": normalized_message,
        }
    return {
        "index": default_index,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": str(choice)}],
        },
    }


def _normalize_message(message: Dict[str, Any], default_role: str = "user") -> Dict[str, Any]:
    role = message.get("role") or default_role
    normalized = {"role": role}
    content = message.get("content")
    normalized_content: List[Dict[str, Any]] = []
    if isinstance(content, list):
        normalized_content = [_normalize_content(fragment) for fragment in content if fragment is not None]
    elif isinstance(content, dict):
        normalized_content = [_normalize_content(content)]
    elif isinstance(content, str):
        normalized_content = [{"type": "text", "text": content}]
    elif content is None and message.get("text") is not None:
        normalized_content = [{"type": "text", "text": str(message.get("text"))}]

    normalized["content"] = [fragment for fragment in normalized_content if fragment]

    for extra_key in ("tool_calls", "tool_use", "model_output", "path", "name"):
        if extra_key in message:
            normalized[extra_key] = message[extra_key]
    return normalized


def _normalize_content(fragment: Any) -> Dict[str, Any]:
    if isinstance(fragment, str):
        return {"type": "text", "text": fragment}
    if not isinstance(fragment, dict):
        return {"type": "text", "text": str(fragment)}

    fragment_type = fragment.get("type")
    if fragment_type == "text":
        return {"type": "text", "text": str(fragment.get("text", ""))}
    if fragment_type in {"image", "audio", "video"}:
        payload = fragment.get(fragment_type)
        if payload is None and "data" in fragment:
            payload = fragment.get("data")
        if payload is None and "url" in fragment:
            payload = {"url": fragment["url"]}
        normalized = {"type": fragment_type}
        if payload is not None:
            normalized[fragment_type] = payload
        if "format" in fragment:
            normalized["format"] = fragment["format"]
        return normalized
    if fragment_type in {"image_url", "audio_url", "video_url", "file_url"}:
        key = fragment_type
        url_payload = fragment.get(key) or fragment.get("url")
        if isinstance(url_payload, dict):
            return {"type": fragment_type, fragment_type: url_payload}
        return {"type": fragment_type, fragment_type: {"url": str(url_payload)}}
    if "url" in fragment and fragment_type:
        return {"type": fragment_type, fragment_type: {"url": fragment["url"]}}

    if "text" in fragment and fragment_type is None:
        return {"type": "text", "text": str(fragment["text"])}
    return {"type": "text", "text": json.dumps(fragment, ensure_ascii=False)}


def _inject_modal_fragments(sample: Dict[str, Any], messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    visual_fragments = _normalize_modal_list(sample.get("visual"), modal_type="image")
    audio_fragments = _normalize_modal_list(sample.get("audio"), modal_type="audio")
    attachments = visual_fragments + audio_fragments

    if not attachments:
        return messages

    normalized_messages = list(messages)
    target_message = next((msg for msg in normalized_messages if msg.get("role") == "user"), None)
    if target_message is None:
        target_message = {"role": "user", "content": []}
        normalized_messages.append(target_message)

    content_list = list(target_message.get("content") or [])
    content_list.extend(attachments)
    target_message["content"] = content_list
    return normalized_messages


def _normalize_modal_list(value: Any, modal_type: str) -> List[Dict[str, Any]]:
    items: Sequence[Any]
    if value is None:
        items = ()
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        items = (value,)

    fragments: List[Dict[str, Any]] = []
    for item in items:
        fragment = _make_modal_fragment(item, modal_type)
        if fragment:
            fragments.append(fragment)
    return fragments


def _make_modal_fragment(item: Any, modal_type: str) -> Dict[str, Any]:
    if item is None:
        return {}
    if isinstance(item, dict):
        fragment = dict(item)
        fragment.setdefault("type", modal_type)
        return _normalize_content(fragment)
    if _is_pil_image(item):
        return {"type": modal_type, modal_type: item}
    if isinstance(item, (bytes, bytearray)):
        return {"type": modal_type, modal_type: {"data": item}}
    if isinstance(item, str):
        if item.startswith("http://") or item.startswith("https://"):
            if modal_type == "image":
                return {"type": "image_url", "image_url": {"url": item}}
            if modal_type == "audio":
                return {"type": "audio_url", "audio_url": {"url": item}}
            if modal_type == "video":
                return {"type": "video_url", "video_url": {"url": item}}
        return {"type": modal_type, modal_type: {"url": item}}
    return {"type": modal_type, modal_type: {"data": item}}


def _is_pil_image(obj: Any) -> bool:
    if _PILImage is None:
        return False
    if not isinstance(_PILImage, type):
        return False
    return isinstance(obj, _PILImage)  # type: ignore[arg-type]


__all__ = [
    "ensure_sample_id",
    "normalize_messages",
    "normalize_choices",
    "normalize_sample",
    "ensure_chat_template_flags",
    "get_prompt",
    "list_images",
]
