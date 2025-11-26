"""Backends for HuggingFace serverless inference & dedicated endpoints."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.registry import registry
from gage_eval.role.model.config.hf_inference import (
    HFServerlessBackendConfig,
    HFInferenceEndpointBackendConfig,
)


@registry.asset(
    "backends",
    "hf_serverless",
    desc="HuggingFace Inference API（Serverless）",
    tags=("llm", "remote", "hf"),
    modalities=("text",),
    config_schema_ref="gage_eval.role.model.config.hf_inference:HFServerlessBackendConfig",
)
class HFServerlessBackend(EngineBackend):
    """Wrapper around huggingface_hub.InferenceClient."""

    def load_model(self, config_dict: Dict[str, Any]):
        try:  # pragma: no cover - optional dependency
            from huggingface_hub import InferenceClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("huggingface_hub is required for hf_serverless backend") from exc

        self.config = HFServerlessBackendConfig(**config_dict)
        token = config_dict.get("huggingface_token") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self._client = InferenceClient(model=self.config.model_name, token=token, timeout=self.config.timeout)
        self._base_params = _generation_dict(self.config.generation_parameters)
        self._max_retries = self.config.max_retries
        self._wait_for_model = self.config.wait_for_model
        self._headers = dict(self.config.extra_headers or {})
        return self._client

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = inputs.get("prompt") or ""
        sampling_params = inputs.get("sampling_params") or {}
        stop_sequences = _extract_stop_sequences(inputs)
        params = self._prepare_generation_kwargs(sampling_params, stop_sequences)
        attempt = 0
        while True:
            try:
                response = self._client.text_generation(
                    prompt=prompt,
                    wait_for_model=self._wait_for_model,
                    extra_headers=self._headers or None,
                    **params,
                )
                text = _extract_hf_text(response)
                return {"answer": text, "raw_response": response}
            except Exception as exc:  # pragma: no cover - network failures
                attempt += 1
                if attempt > self._max_retries:
                    raise
                logger.warning("HFServerlessBackend request failed (attempt {}): {}", attempt, exc)
                time.sleep(min(2**attempt, 5))

    def _prepare_generation_kwargs(self, sampling_params: Dict[str, Any], stop_sequences: Optional[List[str]]):
        params = dict(self._base_params)
        overrides = {
            "max_new_tokens": sampling_params.get("max_new_tokens"),
            "temperature": sampling_params.get("temperature"),
            "top_p": sampling_params.get("top_p"),
            "top_k": sampling_params.get("top_k"),
            "repetition_penalty": sampling_params.get("repetition_penalty"),
            "seed": sampling_params.get("seed"),
        }
        params.update({k: v for k, v in overrides.items() if v is not None})
        if stop_sequences:
            params["stop"] = stop_sequences
        params.setdefault("max_new_tokens", 128)
        params.setdefault("details", False)
        params.setdefault("return_full_text", False)
        return params


@registry.asset(
    "backends",
    "hf_inference_endpoint",
    desc="HuggingFace Inference Endpoint（Dedicated TGI）",
    tags=("llm", "remote", "hf"),
    modalities=("text",),
    config_schema_ref="gage_eval.role.model.config.hf_inference:HFInferenceEndpointBackendConfig",
)
class HFInferenceEndpointBackend(EngineBackend):
    """Backend that ensures a dedicated inference endpoint is running."""

    def load_model(self, config_dict: Dict[str, Any]):
        try:  # pragma: no cover - optional dependency
            from huggingface_hub import (
                InferenceEndpoint,
                InferenceEndpointError,
                HfHubHTTPError,
                create_inference_endpoint,
                get_inference_endpoint,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("huggingface_hub is required for hf_inference_endpoint backend") from exc

        self._hf = {
            "InferenceEndpoint": InferenceEndpoint,
            "InferenceEndpointError": InferenceEndpointError,
            "HfHubHTTPError": HfHubHTTPError,
            "create_inference_endpoint": create_inference_endpoint,
            "get_inference_endpoint": get_inference_endpoint,
        }
        self.config = HFInferenceEndpointBackendConfig(**config_dict)
        self._token = self.config.huggingface_token or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not self._token:
            raise ValueError("hf_inference_endpoint backend requires a HuggingFace token via config or env")
        self._base_params = _generation_dict(self.config.generation_parameters)
        self._endpoint = self._ensure_endpoint()
        self._client = self._endpoint.client
        self._async_client = getattr(self._endpoint, "async_client", None)
        self._created = not self.config.reuse_existing
        return self._client

    def _ensure_endpoint(self):
        cfg = self.config
        name = cfg.normalized_endpoint_name()
        endpoint = None
        if cfg.reuse_existing:
            try:
                endpoint = self._hf["get_inference_endpoint"](name=name, namespace=cfg.namespace, token=self._token)
            except self._hf["HfHubHTTPError"]:
                endpoint = None
        if endpoint is None and cfg.auto_start:
            if not cfg.model_name:
                raise ValueError("auto_start=True 时必须提供 model_name 以创建 endpoint")
            logger.info("Creating HuggingFace Inference Endpoint '{}' (model={})", name, cfg.model_name)
            env = {
                "HF_MODEL_TRUST_REMOTE_CODE": "true",
                **cfg.env_vars,
            }
            if cfg.dtype:
                env["HF_INFERENCE_PRECISION"] = cfg.dtype
            image_url = cfg.image_url or "ghcr.io/huggingface/text-generation-inference:3.0.1"
            endpoint = self._hf["create_inference_endpoint"](
                name=name,
                namespace=cfg.namespace,
                repository=cfg.model_name,
                revision=cfg.revision,
                framework=cfg.framework,
                task="text-generation",
                accelerator=cfg.accelerator,
                vendor=cfg.vendor,
                region=cfg.region,
                type=cfg.endpoint_type,
                instance_type=cfg.instance_type or "nvidia-a10g",
                instance_size=cfg.instance_size or "x1",
                custom_image={"env": env, "health_route": "/health", "url": image_url},
                token=self._token,
            )
            self.config.reuse_existing = False
        if endpoint is None:
            raise ValueError(f"Endpoint '{name}' 不存在，且未启用 auto_start")
        logger.info("Waiting for endpoint '{}' to be running…", name)
        endpoint.wait(timeout=cfg.wait_timeout, refresh_every=cfg.poll_interval)
        if endpoint.status != "running":
            raise RuntimeError(f"Endpoint '{name}' 未在 {cfg.wait_timeout}s 内就绪，当前状态: {endpoint.status}")
        return endpoint

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = inputs.get("prompt") or ""
        stop_sequences = _extract_stop_sequences(inputs)
        sampling_params = inputs.get("sampling_params") or {}
        params = self._compose_generation_kwargs(sampling_params, stop_sequences)
        response = self._client.text_generation(prompt=prompt, wait_for_model=True, **params)
        return {"answer": _extract_hf_text(response), "raw_response": response}

    def close(self) -> None:
        if self._endpoint and self.config.delete_on_exit and not self.config.reuse_existing:
            try:
                self._endpoint.delete()
                logger.info("Deleted inference endpoint '{}'", self._endpoint.name)
            except Exception as exc:  # pragma: no cover - clean-up best effort
                logger.warning("Failed to delete endpoint {}: {}", self._endpoint.name, exc)

    def _compose_generation_kwargs(self, sampling_params: Dict[str, Any], stop_sequences: Optional[List[str]]):
        params = dict(self._base_params)
        overrides = {
            "max_new_tokens": sampling_params.get("max_new_tokens"),
            "temperature": sampling_params.get("temperature"),
            "top_p": sampling_params.get("top_p"),
            "top_k": sampling_params.get("top_k"),
            "repetition_penalty": sampling_params.get("repetition_penalty"),
        }
        params.update({k: v for k, v in overrides.items() if v is not None})
        if stop_sequences:
            params["stop"] = stop_sequences
        params.setdefault("details", False)
        params.setdefault("decoder_input_details", False)
        params.setdefault("return_full_text", False)
        params.setdefault("max_new_tokens", 256)
        return params


def _generation_dict(params):
    data = params.to_dict()
    allowed = ("max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "min_p")
    return {k: v for k, v in data.items() if k in allowed and v is not None}


def _extract_stop_sequences(inputs: Dict[str, Any]) -> List[str]:
    stop = inputs.get("sampling_params", {}).get("stop")
    if not stop:
        sample = inputs.get("sample") or {}
        stop = sample.get("sampling_params", {}).get("stop")
    if not stop:
        return []
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [s for s in stop if isinstance(s, str)]
    return []


def _extract_hf_text(response: Any) -> str:
    if hasattr(response, "generated_text"):
        return response.generated_text
    if isinstance(response, dict) and "generated_text" in response:
        return response["generated_text"]
    if isinstance(response, list) and response and isinstance(response[0], dict):
        return response[0].get("generated_text") or response[0].get("text", "")
    return str(response)
