"""Normalize multimodal chat message blocks for LiteLLM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, NoReturn, Sequence

from gage_eval.role.model.backends.litellm.errors import (
    INVALID_MEDIA_PAYLOAD,
    MEDIA_PAYLOAD_TOO_LARGE,
)
from gage_eval.role.model.backends.litellm.service_profile import VLLMServiceProfile
from gage_eval.role.model.config.litellm import LiteLLMMultimodalPolicyConfig


_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_BASE64_ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")


@dataclass(frozen=True)
class _DataUrl:
    mime_type: str
    payload: str
    decoded_size: int


class MultimodalMessageNormalizer:
    """Preserve LiteLLM-supported multimodal content blocks without silent drops."""

    _MULTIMODAL_TYPES = frozenset(
        {
            "image",
            "image_url",
            "input_audio",
            "audio_url",
            "video_url",
            "file",
            "file_url",
            "document",
        }
    )

    def __init__(
        self,
        policy: LiteLLMMultimodalPolicyConfig | Mapping[str, Any] | None = None,
        *,
        service_profile: VLLMServiceProfile | None = None,
    ) -> None:
        self.policy = self._coerce_policy(policy)
        self.service_profile = service_profile

    def normalize(
        self,
        messages: Sequence[Mapping[str, Any]] | None,
        *,
        multimodal: LiteLLMMultimodalPolicyConfig | Mapping[str, Any] | None = None,
        service_profile: VLLMServiceProfile | None = None,
        profile: VLLMServiceProfile | None = None,
        preserve_blocks: bool | None = None,
        audio_url_mapping: str | None = None,
        default_audio_format: str | None = None,
        max_media_bytes: int | None = None,
        fallback: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Return normalized messages suitable for LiteLLM ``completion`` calls."""

        policy = self._resolve_policy(
            multimodal,
            preserve_blocks=preserve_blocks,
            audio_url_mapping=audio_url_mapping,
            default_audio_format=default_audio_format,
            max_media_bytes=max_media_bytes,
            fallback=fallback,
        )
        normalized: List[Dict[str, Any]] = []
        for message in messages or []:
            new_message = dict(message)
            content = message.get("content")
            if isinstance(content, list):
                new_message["content"] = self._normalize_content_list(content, policy)
            normalized.append(new_message)

        resolved_profile = service_profile or profile or self.service_profile
        if self._is_language_model_only(resolved_profile):
            offending_type = self._first_multimodal_type(normalized)
            if offending_type:
                raise ValueError(
                    f"language_model_only: multimodal content block type={offending_type} is not supported "
                    "(error_type=unsupported_capability)"
                )
        return normalized

    def _normalize_content_list(
        self,
        content: Sequence[Any],
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in content:
            block = self._normalize_content_item(item, policy)
            if block is not None:
                normalized.append(block)
        return normalized

    def _normalize_content_item(
        self,
        item: Any,
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> Dict[str, Any] | None:
        if item is None:
            return None
        if not isinstance(item, Mapping):
            return {"type": "text", "text": str(item)}

        block_type = item.get("type")
        if block_type is None and "text" in item:
            block_type = "text"

        if block_type == "text":
            return self._normalize_text(item)
        if block_type in self._MULTIMODAL_TYPES and policy.preserve_blocks is False:
            return self._handle_unsupported_block(item, block_type, policy)
        if block_type == "image_url":
            return self._normalize_image_url(item, policy)
        if block_type == "image":
            return self._normalize_image(item, policy)
        if block_type == "input_audio":
            return self._normalize_input_audio(item, policy)
        if block_type == "audio_url":
            return self._normalize_audio_url(item, policy)
        if block_type == "video_url":
            return self._normalize_video_url(item, policy)
        if block_type == "file":
            return self._normalize_file(item, policy)
        if block_type == "file_url":
            return self._normalize_file_url(item, policy)
        if block_type == "document":
            return self._normalize_document(item, policy)
        return self._handle_unsupported_block(item, block_type, policy)

    @staticmethod
    def _normalize_text(item: Mapping[str, Any]) -> Dict[str, str]:
        return {"type": "text", "text": str(item.get("text", ""))}

    def _normalize_image_url(
        self,
        item: Mapping[str, Any],
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> Dict[str, Any]:
        payload = self._url_payload_from_block(item, "image_url", aliases=("image",), policy=policy)
        return {"type": "image_url", "image_url": payload}

    def _normalize_image(
        self,
        item: Mapping[str, Any],
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> Dict[str, Any]:
        payload = self._url_payload_from_block(item, "image", aliases=("image_url",), policy=policy)
        return {"type": "image_url", "image_url": payload}

    def _normalize_input_audio(
        self,
        item: Mapping[str, Any],
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> Dict[str, Any]:
        raw = item.get("input_audio")
        payload = dict(raw) if isinstance(raw, Mapping) else {}
        for key in ("data", "url", "format"):
            if item.get(key) is not None and key not in payload:
                payload[key] = item[key]
        if not payload.get("data") and not payload.get("url"):
            self._raise_invalid("input_audio", "missing data or url")
        data = payload.get("data")
        if isinstance(data, str):
            self._validate_base64_payload(data, "input_audio", policy)
        url = payload.get("url")
        if isinstance(url, str) and url.startswith("data:"):
            self._parse_data_url(url, "input_audio", policy)
        return {"type": "input_audio", "input_audio": payload}

    def _normalize_audio_url(
        self,
        item: Mapping[str, Any],
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> Dict[str, Any]:
        payload = self._url_payload_from_block(item, "audio_url", aliases=("audio",), policy=policy)
        if policy.audio_url_mapping == "input_audio":
            url = payload.get("url")
            if not isinstance(url, str) or not url.startswith("data:"):
                self._raise_invalid("audio_url", "audio_url_mapping=input_audio requires a data URL")
            data_url = self._parse_data_url(url, "audio_url", policy)
            audio_format = self._audio_format_from_mime(data_url.mime_type)
            audio_format = audio_format or payload.get("format") or policy.default_audio_format
            if not audio_format:
                self._raise_invalid("audio_url", "missing audio format")
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": data_url.payload,
                    "format": str(audio_format),
                },
            }
        return {"type": "audio_url", "audio_url": payload}

    def _normalize_video_url(
        self,
        item: Mapping[str, Any],
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> Dict[str, Any]:
        payload = self._url_payload_from_block(item, "video_url", aliases=("video",), policy=policy)
        return {"type": "video_url", "video_url": payload}

    def _normalize_file(
        self,
        item: Mapping[str, Any],
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> Dict[str, Any]:
        raw = item.get("file")
        payload = self._file_payload(raw, item, policy)
        return {"type": "file", "file": payload}

    def _normalize_file_url(
        self,
        item: Mapping[str, Any],
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> Dict[str, Any]:
        raw = item.get("file_url")
        payload = self._file_url_payload(raw, item, policy)
        return {"type": "file", "file": payload}

    def _normalize_document(
        self,
        item: Mapping[str, Any],
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> Dict[str, Any]:
        source = item.get("source")
        if isinstance(source, Mapping):
            source_payload = dict(source)
            self._validate_document_payload(source_payload, policy)
            return {"type": "document", "source": source_payload}
        if isinstance(source, str):
            self._validate_document_payload(source, policy)
            return {"type": "document", "source": source}

        raw = item.get("document")
        payload: Any
        if isinstance(raw, Mapping):
            payload = dict(raw)
        elif isinstance(raw, str):
            payload = raw
        else:
            url = item.get("url")
            if isinstance(url, str):
                payload = {"url": url}
            else:
                self._raise_invalid("document", "missing document source")
        self._validate_document_payload(payload, policy)
        return {"type": "document", "document": payload}

    def _url_payload_from_block(
        self,
        item: Mapping[str, Any],
        field_name: str,
        *,
        aliases: Sequence[str],
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> Dict[str, Any]:
        raw = item.get(field_name)
        if isinstance(raw, Mapping):
            payload = dict(raw)
        elif isinstance(raw, str):
            payload = {"url": raw}
        else:
            payload = {}

        for key in ("url", *aliases):
            value = item.get(key)
            if value and "url" not in payload:
                payload["url"] = value
        for key in ("detail", "format"):
            if item.get(key) is not None and key not in payload:
                payload[key] = item[key]

        url = payload.get("url")
        if not isinstance(url, str) or not url:
            self._raise_invalid(str(field_name), "missing url")
        if url.startswith("data:"):
            self._parse_data_url(url, str(field_name), policy)
        return payload

    def _file_payload(
        self,
        raw: Any,
        item: Mapping[str, Any],
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> Dict[str, Any]:
        if isinstance(raw, Mapping):
            payload = dict(raw)
        elif isinstance(raw, str):
            payload = self._file_payload_from_string(raw, policy)
        else:
            payload = {}

        for key in ("file_id", "file_data", "url", "filename"):
            if item.get(key) is not None and key not in payload:
                payload[key] = item[key]

        if not any(payload.get(key) for key in ("file_id", "file_data", "url")):
            self._raise_invalid("file", "missing file_id, file_data, or url")
        file_data = payload.get("file_data")
        if isinstance(file_data, str):
            if file_data.startswith("data:"):
                self._parse_data_url(file_data, "file", policy)
            else:
                self._validate_base64_payload(file_data, "file", policy)
        url = payload.get("url")
        if isinstance(url, str) and url.startswith("data:"):
            self._parse_data_url(url, "file", policy)
        return payload

    def _file_url_payload(
        self,
        raw: Any,
        item: Mapping[str, Any],
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> Dict[str, Any]:
        if isinstance(raw, Mapping):
            payload = dict(raw)
            url = payload.pop("url", None)
            if isinstance(url, str) and url.startswith("data:"):
                self._parse_data_url(url, "file", policy)
            if payload.get("file_id") is None and isinstance(url, str):
                payload["file_id"] = url
        elif isinstance(raw, str):
            payload = {"file_id": raw}
        else:
            payload = {}

        item_url = item.get("url")
        if item.get("file_id") is not None and "file_id" not in payload:
            payload["file_id"] = item["file_id"]
        if isinstance(item_url, str) and item_url.startswith("data:"):
            self._parse_data_url(item_url, "file", policy)
        if item_url is not None and "file_id" not in payload:
            payload["file_id"] = item_url
        for key in ("file_data", "filename"):
            if item.get(key) is not None and key not in payload:
                payload[key] = item[key]

        if not any(payload.get(key) for key in ("file_id", "file_data")):
            self._raise_invalid("file", "missing file_id or file_data")
        file_data = payload.get("file_data")
        if isinstance(file_data, str):
            if file_data.startswith("data:"):
                self._parse_data_url(file_data, "file", policy)
            else:
                self._validate_base64_payload(file_data, "file", policy)
        file_id = payload.get("file_id")
        if isinstance(file_id, str) and file_id.startswith("data:"):
            self._parse_data_url(file_id, "file", policy)
        return payload

    def _file_payload_from_string(self, value: str, policy: LiteLLMMultimodalPolicyConfig) -> Dict[str, Any]:
        if value.startswith("data:"):
            self._parse_data_url(value, "file", policy)
            return {"file_data": value}
        if _URL_SCHEME_RE.match(value):
            return {"url": value}
        return {"file_id": value}

    def _validate_document_payload(
        self,
        payload: Any,
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> None:
        if isinstance(payload, Mapping):
            self._validate_document_mapping(payload, policy)
        elif isinstance(payload, str) and payload.startswith("data:"):
            self._parse_data_url(payload, "document", policy)

    def _validate_document_mapping(
        self,
        payload: Mapping[str, Any],
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> None:
        for key, value in payload.items():
            if isinstance(value, Mapping):
                self._validate_document_mapping(value, policy)
            elif isinstance(value, str) and value.startswith("data:"):
                self._parse_data_url(value, "document", policy)
            elif key in {"data", "file_data"} and isinstance(value, str):
                self._validate_base64_payload(value, "document", policy)

    def _handle_unsupported_block(
        self,
        item: Mapping[str, Any],
        block_type: Any,
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> Dict[str, Any]:
        if policy.fallback == "preserve":
            return dict(item)
        if policy.fallback == "text":
            text = item.get("text")
            if text is None:
                text = f"[unsupported multimodal block: {block_type or 'unknown'}]"
            return {"type": "text", "text": str(text)}
        raise ValueError(
            f"unsupported_modality: type={block_type or 'unknown'} fallback=fail "
            "(error_type=unsupported_capability)"
        )

    def _parse_data_url(
        self,
        value: str,
        block_type: str,
        policy: LiteLLMMultimodalPolicyConfig,
    ) -> _DataUrl:
        comma_index = value.find(",")
        if not value.startswith("data:") or comma_index < len("data:"):
            self._raise_invalid(block_type, "invalid data URL")
        mime_type, has_base64 = self._parse_data_url_metadata(value, len("data:"), comma_index)
        if not has_base64:
            self._raise_invalid(block_type, "data URL must use base64 payload")
        payload_start = comma_index + 1
        decoded_size = self._validate_base64_payload_range(
            value,
            payload_start,
            len(value),
            block_type,
            policy,
            mime_type=mime_type,
        )
        payload = value[payload_start:]
        return _DataUrl(mime_type=mime_type, payload=payload, decoded_size=decoded_size)

    @staticmethod
    def _parse_data_url_metadata(value: str, start: int, end: int) -> tuple[str, bool]:
        mime_type = ""
        has_base64 = False
        part_start = start
        part_index = 0
        index = start
        while index <= end:
            if index == end or value[index] == ";":
                if index > part_start:
                    if part_index == 0 and value.find("/", part_start, index) != -1:
                        mime_type = _slice_mime_type(value, part_start, index)
                    elif _range_equals_ascii_lower(value, part_start, index, "base64"):
                        has_base64 = True
                part_index += 1
                part_start = index + 1
            index += 1
        return mime_type, has_base64

    def _validate_base64_payload(
        self,
        payload: str,
        block_type: str,
        policy: LiteLLMMultimodalPolicyConfig,
        *,
        mime_type: str | None = None,
    ) -> int:
        return self._validate_base64_payload_range(
            payload,
            0,
            len(payload),
            block_type,
            policy,
            mime_type=mime_type,
        )

    def _validate_base64_payload_range(
        self,
        value: str,
        start: int,
        end: int,
        block_type: str,
        policy: LiteLLMMultimodalPolicyConfig,
        *,
        mime_type: str | None = None,
    ) -> int:
        encoded_chars = 0
        padding = 0
        padding_started = False
        limit = policy.max_media_bytes

        for index in range(start, end):
            char = value[index]
            if char.isspace():
                continue
            if char == "=":
                padding_started = True
                padding += 1
                encoded_chars += 1
                if padding > 2:
                    self._raise_invalid(block_type, "invalid base64 padding", mime_type=mime_type)
            elif char in _BASE64_ALPHABET:
                if padding_started:
                    self._raise_invalid(block_type, "invalid base64 padding", mime_type=mime_type)
                encoded_chars += 1
            else:
                self._raise_invalid(block_type, "invalid base64 payload", mime_type=mime_type)

            if limit is not None:
                decoded_lower_bound = max(0, (encoded_chars // 4) * 3 - 2)
                if decoded_lower_bound > limit:
                    raise self._media_payload_too_large(
                        block_type,
                        size=decoded_lower_bound,
                        limit=limit,
                        mime_type=mime_type,
                    )

        if encoded_chars == 0 or encoded_chars % 4 != 0:
            self._raise_invalid(block_type, "invalid base64 payload", mime_type=mime_type)

        decoded_size = (encoded_chars // 4) * 3 - padding
        if limit is not None and decoded_size > limit:
            raise self._media_payload_too_large(
                block_type,
                size=decoded_size,
                limit=limit,
                mime_type=mime_type,
            )
        return decoded_size

    @staticmethod
    def _audio_format_from_mime(mime_type: str) -> str | None:
        if not mime_type.startswith("audio/"):
            return None
        subtype = mime_type.split("/", 1)[1].strip()
        return subtype or None

    @staticmethod
    def _is_language_model_only(profile: VLLMServiceProfile | None) -> bool:
        if profile is None:
            return False
        if profile.language_model_only is True:
            return True
        topology = profile.topology
        return isinstance(topology, Mapping) and topology.get("language_model_only") is True

    @classmethod
    def _first_multimodal_type(cls, messages: Sequence[Mapping[str, Any]]) -> str | None:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, Mapping) and item.get("type") in cls._MULTIMODAL_TYPES:
                    return str(item.get("type"))
        return None

    @staticmethod
    def _coerce_policy(
        policy: LiteLLMMultimodalPolicyConfig | Mapping[str, Any] | None,
    ) -> LiteLLMMultimodalPolicyConfig:
        if policy is None:
            return LiteLLMMultimodalPolicyConfig()
        if isinstance(policy, LiteLLMMultimodalPolicyConfig):
            return policy
        return LiteLLMMultimodalPolicyConfig(**dict(policy))

    def _resolve_policy(
        self,
        policy: LiteLLMMultimodalPolicyConfig | Mapping[str, Any] | None,
        *,
        preserve_blocks: bool | None,
        audio_url_mapping: str | None,
        default_audio_format: str | None,
        max_media_bytes: int | None,
        fallback: str | None,
    ) -> LiteLLMMultimodalPolicyConfig:
        resolved = self._coerce_policy(policy) if policy is not None else self.policy
        overrides: Dict[str, Any] = {}
        if preserve_blocks is not None:
            overrides["preserve_blocks"] = preserve_blocks
        if audio_url_mapping is not None:
            overrides["audio_url_mapping"] = audio_url_mapping
        if default_audio_format is not None:
            overrides["default_audio_format"] = default_audio_format
        if max_media_bytes is not None:
            overrides["max_media_bytes"] = max_media_bytes
        if fallback is not None:
            overrides["fallback"] = fallback
        if not overrides:
            return resolved
        if hasattr(resolved, "model_dump"):
            data = resolved.model_dump()
        else:  # pragma: no cover - pydantic v1 fallback
            data = resolved.dict()
        data.update(overrides)
        return LiteLLMMultimodalPolicyConfig(**data)

    @staticmethod
    def _raise_invalid(block_type: str, reason: str, *, mime_type: str | None = None) -> NoReturn:
        mime = f" mime={_safe_diag_value(mime_type)}" if mime_type else ""
        raise ValueError(
            f"{INVALID_MEDIA_PAYLOAD}: type={block_type} reason={reason}{mime} "
            f"(error_type={INVALID_MEDIA_PAYLOAD})"
        )

    @staticmethod
    def _media_payload_too_large(
        block_type: str,
        *,
        size: int,
        limit: int,
        mime_type: str | None,
    ) -> ValueError:
        mime = f" mime={_safe_diag_value(mime_type)}" if mime_type else ""
        return ValueError(
            f"{MEDIA_PAYLOAD_TOO_LARGE}: type={block_type} size={size} limit={limit}{mime} "
            f"(error_type={MEDIA_PAYLOAD_TOO_LARGE})"
        )


def _safe_diag_value(value: Any, limit: int = 64) -> str:
    text = str(value)
    safe = "".join(char if 32 <= ord(char) < 127 else "?" for char in text[:limit])
    if len(text) <= limit:
        return safe
    return f"{safe}...(len={len(text)})"


def _slice_mime_type(value: str, start: int, end: int, limit: int = 128) -> str:
    if end - start > limit:
        return ""
    return value[start:end]


def _range_equals_ascii_lower(value: str, start: int, end: int, expected: str) -> bool:
    if end - start != len(expected):
        return False
    for offset, expected_char in enumerate(expected):
        if value[start + offset].lower() != expected_char:
            return False
    return True
