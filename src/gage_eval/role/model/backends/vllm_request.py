"""Shared helpers for normalizing vLLM backend requests."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VLLMRequest:
    sample: Dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    prompt_meta: Dict[str, Any] = field(default_factory=dict)
    cache_namespace: Optional[str] = None
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    output_type: str = "text"


def _normalize_payload_base(payload: Dict[str, Any]) -> VLLMRequest:
    """Normalize common request fields for backend consumption."""

    sample = payload.get("sample") or {}
    messages = payload.get("messages")
    if messages is None:
        messages = sample.get("messages") or []
    inputs = payload.get("inputs") or sample.get("inputs") or {}
    prompt = (
        payload.get("prompt")
        or payload.get("text")
        or sample.get("prompt")
        or sample.get("text")
        or ""
    )
    prompt_meta = payload.get("prompt_meta") or {}
    cache_namespace = payload.get("cache_namespace") or sample.get("cache_namespace")
    chat_template_kwargs = payload.get("chat_template_kwargs") or sample.get("chat_template_kwargs") or {}
    output_type = payload.get("output_type") or sample.get("output_type") or "text"

    sampling_params: Dict[str, Any] = {}
    for source in (
        sample.get("generation_params") if isinstance(sample, dict) else None,
        payload.get("generation_params"),
        sample.get("sampling_params"),
        payload.get("sampling_params"),
    ):
        if isinstance(source, dict):
            sampling_params.update(source)

    return VLLMRequest(
        sample=sample if isinstance(sample, dict) else {},
        prompt=str(prompt),
        messages=list(messages) if isinstance(messages, list) else [],
        inputs=inputs if isinstance(inputs, dict) else {},
        sampling_params=sampling_params,
        prompt_meta=prompt_meta if isinstance(prompt_meta, dict) else {},
        cache_namespace=str(cache_namespace) if cache_namespace is not None else None,
        chat_template_kwargs=chat_template_kwargs if isinstance(chat_template_kwargs, dict) else {},
        output_type=str(output_type or "text"),
    )


def normalize_vllm_payload(payload: Dict[str, Any]) -> VLLMRequest:
    """Normalize request fields emitted by TextGenerationMixin and legacy callers."""
    return _normalize_payload_base(payload)


def resolve_sample_n(payload: Dict[str, Any], sampling_params: Optional[Dict[str, Any]], *, default: int = 1) -> int:
    """Resolve sample_n from payload or sampling params."""

    candidate = payload.get("sample_n")
    if candidate is None and sampling_params:
        candidate = sampling_params.get("n") or sampling_params.get("num_samples")
    try:
        value = int(candidate) if candidate is not None else int(default)
    except Exception:
        value = int(default)
    return value if value > 0 else int(default)


@dataclass
class BackendRequestContext(VLLMRequest):
    sample_n: int = 1
    request_id: Optional[str] = None
    chat_meta: Dict[str, Any] = field(default_factory=dict)


def _resolve_request_id(payload: Dict[str, Any], sample: Dict[str, Any], request_prefix: str) -> str:
    """Best-effort request_id resolution with prefix fallback."""

    candidate = payload.get("request_id")
    if not candidate and isinstance(sample, dict):
        for key in ("request_id", "id", "sample_id", "idx"):
            candidate = sample.get(key)
            if candidate:
                break
    if not candidate and isinstance(payload, dict):
        candidate = payload.get("id") or payload.get("sample_id")

    request_id = str(candidate).strip() if candidate is not None else ""
    if not request_id:
        request_id = f"{request_prefix}_{uuid.uuid4().hex}"
    return request_id


def _collect_chat_meta(payload: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
    chat_meta: Dict[str, Any] = {}
    for source in (payload.get("chat_meta"), sample.get("chat_meta") if isinstance(sample, dict) else None):
        if isinstance(source, dict):
            chat_meta.update(source)
    return chat_meta


def normalize_request_payload(
    payload: Dict[str, Any],
    *,
    request_prefix: Optional[str] = "backend",
    default_sample_n: int = 1,
) -> BackendRequestContext:
    """Generic normalization that also resolves request_id/sample_n/chat_meta."""

    base = _normalize_payload_base(payload)
    chat_meta = _collect_chat_meta(payload, base.sample)
    sample_n = resolve_sample_n(payload, base.sampling_params, default=default_sample_n)
    request_id = _resolve_request_id(payload, base.sample, request_prefix) if request_prefix else None

    return BackendRequestContext(
        sample=base.sample,
        prompt=base.prompt,
        messages=base.messages,
        inputs=base.inputs,
        sampling_params=base.sampling_params,
        prompt_meta=base.prompt_meta,
        cache_namespace=base.cache_namespace,
        chat_template_kwargs=base.chat_template_kwargs,
        output_type=base.output_type,
        sample_n=sample_n,
        request_id=request_id,
        chat_meta=chat_meta,
    )


LegacyVLLMRequest = BackendRequestContext


def normalize_legacy_payload(payload: Dict[str, Any], *, request_prefix: str = "legacy_vllm") -> LegacyVLLMRequest:
    """Legacy wrapper that adds request_id/sample_n/chat_meta on top of normalize_vllm_payload."""

    return normalize_request_payload(payload, request_prefix=request_prefix, default_sample_n=1)
