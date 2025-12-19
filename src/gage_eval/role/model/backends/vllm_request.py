"""Shared helpers for normalizing vLLM backend requests."""

from __future__ import annotations

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


def normalize_vllm_payload(payload: Dict[str, Any]) -> VLLMRequest:
    """Normalize request fields emitted by TextGenerationMixin and legacy callers."""

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
