"""Shared helpers for normalizing vLLM backend requests and vLLM utilities."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


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
    # Collect render flags from payload or sample.metadata/data_tag (Sample no longer carries chat_meta).
    render_keys = ("chat_template_mode", "template_source", "rendered_by", "cache_suffix")
    meta = sample.get("metadata") or sample.get("data_tag") or {} if isinstance(sample, dict) else {}
    for key in render_keys:
        if key in payload:
            chat_meta[key] = payload[key]
        elif isinstance(meta, dict) and key in meta:
            chat_meta[key] = meta[key]
        elif isinstance(sample, dict) and key in sample:
            chat_meta[key] = sample[key]
    # Merge chat_meta from payload or nested metadata if present.
    meta_chat = meta.get("chat_meta") if isinstance(meta, dict) else None
    for source in (payload.get("chat_meta"), meta_chat):
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


def detect_vllm_version() -> Optional[str]:
    """Detect installed vLLM version without hard dependency on the package."""

    try:  # pragma: no cover - optional dependency
        import vllm  # type: ignore
    except Exception:
        logger.warning("vLLM not installed; version detection skipped")
        return None
    ver = getattr(vllm, "__version__", None)
    if isinstance(ver, str):
        return ver
    try:
        from importlib.metadata import version as meta_version

        return meta_version("vllm")
    except Exception:
        return None


def resolve_vllm_mm_support(
    config: Dict[str, Any], vllm_version: Optional[str]
) -> Tuple[bool, str, bool]:
    """Resolve vLLM multimodal support based on version and config flags."""

    strict_mm = bool(
        config.get("strict_multi_modal")
        or os.environ.get("GAGE_EVAL_VLLM_STRICT_MM", "").lower() in {"1", "true", "yes", "on"}
    )
    if config.get("force_prompt_only") or os.environ.get("GAGE_EVAL_VLLM_PROMPT_ONLY", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False, "disabled", strict_mm
    if vllm_version:
        try:
            from packaging import version as _version

            if _version.parse(vllm_version) < _version.parse("0.8.3"):
                logger.warning("vLLM < 0.8.3 detected; defaulting multi_modal_data unsupported")
                return False, "disabled", strict_mm
        except Exception:
            pass
    return True, "inputs", strict_mm


def build_engine_args(config: Dict[str, Any], *, output_type: str, trust_remote_code: bool = True) -> SimpleNamespace:
    """Normalize common engine args for vLLM-like backends."""

    args = SimpleNamespace()
    args.model = config.get("model_path") or config.get("model")
    args.output_type = output_type
    args.trust_remote_code = trust_remote_code
    args.max_length = config.get("max_model_len") or config.get("max_length")
    args.low_vram = bool(config.get("low_vram", False))
    args.block_size = config.get("block_size")
    args.tensor_parallel_size = int(config.get("tensor_parallel_size", config.get("tensor_parallel", 1)))
    args.pipeline_parallel = int(config.get("pipeline_parallel", 1))
    args.pipeline_parallel_size = int(config.get("pipeline_parallel_size", args.pipeline_parallel))
    args.enable_expert_parallel = config.get("enable_expert_parallel")
    args.enforce_eager = config.get("enforce_eager")
    args.num_gpu_blocks = config.get("num_gpu_blocks")
    args.num_cpu_blocks = config.get("num_cpu_blocks")
    args.forced_num_gpu_blocks = config.get("forced_num_gpu_blocks")
    args.num_gpu_blocks_override = config.get("num_gpu_blocks_override")
    return args


def ensure_spawn_start_method() -> None:
    """Force multiprocessing start method to spawn to avoid CUDA re-init issues."""

    try:
        import torch.multiprocessing as mp  # type: ignore
    except Exception:  # pragma: no cover - torch missing
        import multiprocessing as mp  # type: ignore

    current = mp.get_start_method(allow_none=True)
    if current != "spawn":
        try:
            mp.set_start_method("spawn", force=True)
        except Exception:
            logger.warning("backend failed to set multiprocessing start method to spawn")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def resolve_sampling_class():
    """Resolve vLLM SamplingParams if available; otherwise a lightweight placeholder."""

    try:  # pragma: no cover - optional dependency
        from vllm import SamplingParams  # type: ignore

        return SamplingParams
    except Exception:
        class SamplingParams:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

            def __repr__(self) -> str:
                return f"SamplingParams({self.__dict__})"

        return SamplingParams


def build_sampling_params(
    output_type: str,
    runtime_params: Dict[str, Any],
    *,
    default_sampling: Dict[str, Any],
    max_tokens: int,
    sampling_class_resolver=resolve_sampling_class,
):
    """Build sampling params with legacy output_type tweaks (loss/logprobs/max_tokens)."""

    base = dict(default_sampling or {})
    base.update(runtime_params or {})

    if output_type == "loss":
        base.setdefault("temperature", 0)
        base.setdefault("prompt_logprobs", 1)
        base["max_tokens"] = 1
    elif output_type == "prompt_tokens":
        base.setdefault("temperature", 0)
        base.setdefault("prompt_logprobs", base.get("prompt_logprobs", 20))
        base["max_tokens"] = 1
    elif output_type == "next_token_prob":
        base.setdefault("temperature", 0)
        base["max_tokens"] = 1
        top_lp = base.pop("top_logprobs_num", None) or base.get("logprobs")
        if top_lp is not None:
            base["logprobs"] = top_lp
    else:
        base.setdefault("max_tokens", max_tokens)

    sampling_cls = sampling_class_resolver()
    return sampling_cls(**base)
