"""Factory for backend adaptors backed by the global registry."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Type

from loguru import logger
from gage_eval.assets.models.store import resolve_model_handle
from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.registry import registry
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


# Typed + passthrough: 部分后端提供 BackendConfig 模型以声明常用字段和默认值，
# 但仍允许 config 中携带额外字段透传给 Backend 实现。
_BACKEND_CONFIG_SCHEMAS: Dict[str, Type[BackendConfigBase]] = {
    # 文本类 / 远端 HTTP
    "openai_http": OpenAIHTTPBackendConfig,
    "http": HTTPBackendConfig,
    "hf": HFBackendConfig,
    # 多模态 / 本地推理
    "vllm": VLLMBackendConfig,
    "vlm_transformers": VLMTransformersBackendConfig,
    # 检索 / 向量
    "faiss": FaissBackendConfig,
    "flag_embedding": FlagEmbeddingBackendConfig,
    # 语音
    "whisper_asr": WhisperASRBackendConfig,
    # 代理类 HTTP 适配器
    "litellm": LiteLLMBackendConfig,
    "sglang": SGLangBackendConfig,
    "tgi": TGIBackendConfig,
    # 分布式 / 服务器less
    "nanotron": NanotronBackendConfig,
    "hf_serverless": HFServerlessBackendConfig,
    "hf_inference_endpoint": HFInferenceEndpointBackendConfig,
    # 其他
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
    if isinstance(spec, str):
        spec = {"type": spec, "config": {}}
    backend_type = spec.get("type")
    if not backend_type:
        raise ValueError("Backend spec must declare 'type'")
    try:
        backend_cls = registry.get("backends", backend_type)
    except KeyError as exc:
        raise KeyError(f"Backend '{backend_type}' is not registered") from exc
    config = dict(spec.get("config", {}))
    model_id = config.pop("model_id", None)
    if model_id:
        handle = resolve_model_handle(model_id)
        config.setdefault("model_path", handle.local_path)
        config.setdefault("model_metadata", handle.metadata)
    # 若存在对应的 BackendConfig，则先做一次轻量 Schema 解析：
    # - 已声明字段享受类型校验与默认值；
    # - 未声明字段作为 extra 保留，最终仍透传给 Backend。
    cfg_model_cls = _BACKEND_CONFIG_SCHEMAS.get(backend_type)
    if cfg_model_cls is not None:
        try:
            cfg_model = cfg_model_cls(**config)
            # 兼容 Pydantic v1/v2 的导出接口
            if hasattr(cfg_model, "model_dump"):
                typed_config = cfg_model.model_dump()
            else:  # pragma: no cover - v1 fallback
                typed_config = cfg_model.dict()
        except Exception as exc:  # pragma: no cover - 防御性，透出更友好的错误信息
            raise ValueError(f"Invalid config for backend '{backend_type}': {exc}") from exc
    else:
        typed_config = config

    logger.debug("Building backend instance type='{}'", backend_type)
    return backend_cls(typed_config)
