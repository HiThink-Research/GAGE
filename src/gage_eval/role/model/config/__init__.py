"""Pydantic schemas describing backend/runtime parameters."""

from gage_eval.role.model.config.base import BackendConfigBase
from gage_eval.role.model.config.generations import GenerationParameters
from gage_eval.role.model.config.hf import HFBackendConfig
from gage_eval.role.model.config.http import HTTPBackendConfig
from gage_eval.role.model.config.openai_http import OpenAIHTTPBackendConfig
from gage_eval.role.model.config.sglang import SGLangBackendConfig
from gage_eval.role.model.config.tgi import TGIBackendConfig
from gage_eval.role.model.config.vllm import VLLMBackendConfig
from gage_eval.role.model.config.faiss import FaissBackendConfig
from gage_eval.role.model.config.flag_embedding import FlagEmbeddingBackendConfig
from gage_eval.role.model.config.whisper import WhisperASRBackendConfig
from gage_eval.role.model.config.litellm import LiteLLMBackendConfig
from gage_eval.role.model.config.vlm_transformers import VLMTransformersBackendConfig
from gage_eval.role.model.config.nanotron import NanotronBackendConfig, NanotronParallelismConfig
from gage_eval.role.model.config.hf_inference import HFServerlessBackendConfig, HFInferenceEndpointBackendConfig
from gage_eval.role.model.config.inference_providers import InferenceProvidersBackendConfig
from gage_eval.role.model.config.dummy import DummyBackendConfig
from gage_eval.role.model.config.vendor_http import (
    ClaudeBackendConfig,
    GeminiBackendConfig,
    OpenAIBatchBackendConfig,
)

__all__ = [
    "BackendConfigBase",
    "GenerationParameters",
    "HFBackendConfig",
    "HTTPBackendConfig",
    "OpenAIHTTPBackendConfig",
    "SGLangBackendConfig",
    "TGIBackendConfig",
    "VLLMBackendConfig",
    "FaissBackendConfig",
    "FlagEmbeddingBackendConfig",
    "WhisperASRBackendConfig",
    "LiteLLMBackendConfig",
    "VLMTransformersBackendConfig",
    "NanotronBackendConfig",
    "NanotronParallelismConfig",
    "HFServerlessBackendConfig",
    "HFInferenceEndpointBackendConfig",
    "InferenceProvidersBackendConfig",
    "DummyBackendConfig",
    "ClaudeBackendConfig",
    "GeminiBackendConfig",
    "OpenAIBatchBackendConfig",
]
