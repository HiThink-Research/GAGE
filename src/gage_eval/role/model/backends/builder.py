"""Factory for backend adaptors backed by the global registry."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Type

from loguru import logger
from gage_eval.assets.models.store import resolve_model_handle
from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.role.model.config import (
    BackendConfigBase,
    HFBackendConfig,
    HTTPBackendConfig,
    OpenAIHTTPBackendConfig,
    SGLangBackendConfig,
    TGIBackendConfig,
    VLLMBackendConfig,
    FaissBackendConfig,
    FlagEmbeddingBackendConfig,
    WhisperASRBackendConfig,
    LiteLLMBackendConfig,
    VLMTransformersBackendConfig,
    NanotronBackendConfig,
    HFServerlessBackendConfig,
    HFInferenceEndpointBackendConfig,
    DummyBackendConfig,
    ClaudeBackendConfig,
    GeminiBackendConfig,
    OpenAIBatchBackendConfig,
)


# Typed + passthrough: some backends provide a `BackendConfig` model to declare
# common fields and defaults, while still allowing extra config keys to be
# passed through to the backend implementation.
_BACKEND_CONFIG_SCHEMAS: Dict[str, Type[BackendConfigBase]] = {
    # Text / remote HTTP
    "openai_http": OpenAIHTTPBackendConfig,
    "http": HTTPBackendConfig,
    "hf": HFBackendConfig,
    # Multimodal / local inference
    "vllm": VLLMBackendConfig,
    "vlm_transformers": VLMTransformersBackendConfig,
    # Retrieval / embeddings
    "faiss": FaissBackendConfig,
    "flag_embedding": FlagEmbeddingBackendConfig,
    # Speech
    "whisper_asr": WhisperASRBackendConfig,
    # Proxy-style HTTP adapters
    "litellm": LiteLLMBackendConfig,
    "sglang": SGLangBackendConfig,
    "tgi": TGIBackendConfig,
    # Distributed / serverless
    "nanotron": NanotronBackendConfig,
    "hf_serverless": HFServerlessBackendConfig,
    "hf_inference_endpoint": HFInferenceEndpointBackendConfig,
    # Misc
    "dummy": DummyBackendConfig,
    "claude_http": ClaudeBackendConfig,
    "gemini_http": GeminiBackendConfig,
    "openai_batch_http": OpenAIBatchBackendConfig,
}


def register_backend(
    kind: str,
    backend_cls: Type[EngineBackend],
    *,
    desc: str,
    version: str = "v1",
    tags: Optional[Sequence[str]] = None,
    **extra: Any,
) -> None:
    """Compatibility helper for imperative backend registration."""

    from gage_eval.registry import registry

    registry.register(
        "backends",
        kind,
        backend_cls,
        desc=desc,
        version=version,
        tags=tags,
        **extra,
    )
    logger.info("Registered backend '{}' (version={})", kind, version)


def build_backend(spec: Dict[str, Any]) -> EngineBackend:
    """Builds a backend instance from a registry-backed backend spec.

    Args:
        spec: Backend spec. Either a string backend kind (shorthand), or a dict
            with keys:
            - `type`: Backend kind registered under `backends`.
            - `config`: Backend-specific config payload.

    Returns:
        A constructed backend instance.

    Raises:
        ValueError: If the spec is missing required fields, or the config schema
            validation fails.
        KeyError: If the backend kind is not registered.
    """

    # STEP 1: Normalize the spec into a typed dict form.
    if isinstance(spec, str):
        spec = {"type": spec, "config": {}}
    backend_type = spec.get("type")
    if not backend_type:
        raise ValueError("Backend spec must declare 'type'")
    
    from gage_eval.registry import registry
    
    try:
        backend_cls = registry.get("backends", backend_type)
    except KeyError as exc:
        raise KeyError(f"Backend '{backend_type}' is not registered") from exc

    # STEP 2: Prepare config payload (including model-id resolution).
    config = dict(spec.get("config", {}))
    model_id = config.pop("model_id", None)
    if model_id:
        handle = resolve_model_handle(model_id)
        config.setdefault("model_path", handle.local_path)
        config.setdefault("model_metadata", handle.metadata)

    # STEP 3: Apply optional schema validation/normalization (typed + passthrough).
    # Some backends provide a `BackendConfig` model to validate common fields and
    # fill defaults. Unknown fields are kept as extras and still passed through
    # to the backend implementation.
    cfg_model_cls = _BACKEND_CONFIG_SCHEMAS.get(backend_type)
    if cfg_model_cls is not None:
        try:
            cfg_model = cfg_model_cls(**config)
            # NOTE: Support both Pydantic v1 and v2 export APIs.
            if hasattr(cfg_model, "model_dump"):
                typed_config = cfg_model.model_dump()
            else:  # pragma: no cover - v1 fallback
                typed_config = cfg_model.dict()
        except Exception as exc:  # pragma: no cover - defensive: emit actionable errors
            raise ValueError(f"Invalid config for backend '{backend_type}': {exc}") from exc
    else:
        typed_config = config

    # STEP 4: Instantiate the backend class.
    logger.debug("Building backend instance type='{}'", backend_type)
    return backend_cls(typed_config)
