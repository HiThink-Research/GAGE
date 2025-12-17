"""Backends for HuggingFace serverless inference & dedicated endpoints."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import ConnectionError
from loguru import logger

from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.registry import registry
from gage_eval.role.model.config.hf_inference import (
    HFServerlessBackendConfig,
    HFInferenceEndpointBackendConfig,
)

MAX_TIME_FOR_SPINUP = 3600

SORTED_INSTANCE_SIZES = [  # sorted by incremental overall RAM (to load models)
    # type, size
    ("nvidia-a10g", "x1"),
    ("nvidia-t4", "x4"),
    ("nvidia-a100", "x1"),
    ("nvidia-a10g", "x4"),
    ("nvidia-a100", "x2"),
    ("nvidia-a100", "x4"),
]


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
            from huggingface_hub import AsyncInferenceClient, InferenceClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("huggingface_hub is required for hf_serverless backend") from exc

        self.config = HFServerlessBackendConfig(**config_dict)
        token = config_dict.get("huggingface_token") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self._client = InferenceClient(model=self.config.model_name, token=token, timeout=self.config.timeout)
        self._async_client = None
        self._async_semaphore: asyncio.Semaphore | None = None
        env_enable_async = os.environ.get("GAGE_EVAL_ENABLE_ASYNC_HTTP")
        enable_async_cfg = config_dict.get("enable_async", self.config.enable_async)
        self._enable_async = (
            bool(enable_async_cfg)
            if enable_async_cfg is not None
            else (env_enable_async.lower() in {"1", "true", "yes", "on"} if env_enable_async else False)
        )
        max_async = config_dict.get("async_max_concurrency", self.config.async_max_concurrency)
        max_async = 0 if max_async is None else int(max_async)
        if self._enable_async:
            try:
                self._async_client = AsyncInferenceClient(
                    model=self.config.model_name, token=token, timeout=self.config.timeout
                )
                self._async_semaphore = asyncio.Semaphore(max_async) if max_async > 0 else None
            except Exception as exc:  # pragma: no cover - optional dependency path
                logger.warning("Failed to initialize AsyncInferenceClient: {}", exc)
                self._async_client = None
                self._enable_async = False
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

    async def ainvoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._async_client is None or not self._enable_async:
            return await super().ainvoke(payload)

        inputs = self.prepare_inputs(payload)
        prompt = inputs.get("prompt") or ""
        sampling_params = inputs.get("sampling_params") or {}
        stop_sequences = _extract_stop_sequences(inputs)
        params = self._prepare_generation_kwargs(sampling_params, stop_sequences)

        attempt = 0
        start = time.time()
        while True:
            try:
                async def _do_call():
                    return await self._async_client.text_generation(
                        prompt=prompt,
                        wait_for_model=self._wait_for_model,
                        extra_headers=self._headers or None,
                        **params,
                    )

                if self._async_semaphore:
                    async with self._async_semaphore:
                        response = await _do_call()
                else:
                    response = await _do_call()

                text = _extract_hf_text(response)
                return {"answer": text, "raw_response": response, "latency_ms": (time.time() - start) * 1000}
            except Exception as exc:  # pragma: no cover - network/runtime failures
                attempt += 1
                if attempt > self._max_retries:
                    raise
                logger.warning("HFServerlessBackend async request failed (attempt {}): {}", attempt, exc)
                await asyncio.sleep(min(2**attempt, 5))

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
                create_inference_endpoint,
                get_inference_endpoint,
            )
            try:
                from huggingface_hub import HfHubHTTPError  # type: ignore
            except Exception:  # pragma: no cover - compat for older/newer hub versions
                from huggingface_hub.utils import HfHubHTTPError  # type: ignore
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
        env_enable_async = os.environ.get("GAGE_EVAL_ENABLE_ASYNC_HTTP")
        enable_async_cfg = config_dict.get("enable_async", self.config.enable_async)
        self._enable_async = (
            bool(enable_async_cfg)
            if enable_async_cfg is not None
            else (env_enable_async.lower() in {"1", "true", "yes", "on"} if env_enable_async else False)
        )
        max_async = config_dict.get("async_max_concurrency", self.config.async_max_concurrency)
        max_async = 0 if max_async is None else int(max_async)
        self._async_semaphore = asyncio.Semaphore(max_async) if (self._async_client and max_async > 0) else None
        self._created = not self.config.reuse_existing
        return self._client

    def _ensure_endpoint(self):
        cfg = self.config
        name = cfg.normalized_endpoint_name()

        def _resolve_hardware():
            if cfg.instance_type and cfg.instance_size and cfg.vendor and cfg.region:
                return cfg.vendor, cfg.region, cfg.instance_type, cfg.instance_size
            if cfg.model_name:
                try:
                    return self.get_suggested_model_config(cfg.model_name)
                except Exception as exc:  # pragma: no cover - network failures
                    logger.warning("Failed to fetch suggested hardware for {}: {}", cfg.model_name, exc)
            vendor, region = cfg.vendor or "aws", cfg.region or "us-east-1"
            inst_type, inst_size = self.get_larger_hardware_suggestion()
            return vendor, region, inst_type, inst_size

        endpoint = self._try_get_endpoint(name, cfg.namespace) if cfg.reuse_existing else None
        if endpoint is None and not cfg.auto_start:
            raise ValueError(f"Endpoint '{name}' 不存在，且未启用 auto_start")
        if endpoint is None and not cfg.model_name:
            raise ValueError("auto_start=True 时必须提供 model_name 以创建 endpoint")

        vendor, region, instance_type, instance_size = _resolve_hardware()
        must_scaleup_endpoint = False
        timer_start = time.time()

        while (endpoint is None or endpoint.status != "running") and (time.time() - timer_start < MAX_TIME_FOR_SPINUP):
            try:
                if endpoint is None:
                    env_vars = {
                        "MAX_BATCH_PREFILL_TOKENS": "2048",
                        "MAX_INPUT_LENGTH": "2047",
                        "MAX_TOTAL_TOKENS": "2048",
                        "MODEL_ID": "/repository",
                        "HF_MODEL_TRUST_REMOTE_CODE": "true",
                        **self._get_dtype_args(),
                        **self._get_custom_env_vars(),
                    }
                    logger.info("Creating endpoint '{}' (model={})", name, cfg.model_name)
                    endpoint = self._hf["create_inference_endpoint"](
                        name=name,
                        namespace=cfg.namespace,
                        repository=cfg.model_name,
                        revision=cfg.revision,
                        framework=cfg.framework,
                        task="text-generation",
                        accelerator=cfg.accelerator,
                        type=cfg.endpoint_type,
                        vendor=vendor,
                        region=region,
                        instance_size=instance_size,
                        instance_type=instance_type,
                        custom_image={
                            "health_route": "/health",
                            "env": env_vars,
                            "url": (cfg.image_url or "ghcr.io/huggingface/text-generation-inference:3.0.1"),
                        },
                        token=self._token,
                    )
                    self.config.reuse_existing = False  # We created it

                if must_scaleup_endpoint:
                    logger.info("Rescaling endpoint '{}' to {} {}", name, instance_type, instance_size)
                    endpoint.update(instance_size=instance_size, instance_type=instance_type)
                    must_scaleup_endpoint = False

                logger.info("Waiting for endpoint '{}' to be running...", name)
                endpoint.wait(timeout=cfg.wait_timeout, refresh_every=cfg.poll_interval)
                if endpoint.status == "running":
                    return endpoint

            except self._hf["InferenceEndpointError"] as exc:
                logger.warning(
                    "Endpoint failed to start on current hardware ({}, {}) with error {}. Trying to scale up.",
                    instance_type,
                    instance_size,
                    exc,
                )
                instance_type, instance_size = self.get_larger_hardware_suggestion(instance_type, instance_size)
                must_scaleup_endpoint = True

            except self._hf["HfHubHTTPError"] as exc:
                message = str(exc)
                if "Conflict: Quota exceeded" in message or "401 Client Error" in message:
                    raise
                if "Conflict for url" in message:
                    logger.info("Endpoint already exists, switching to reuse_existing=True")
                    endpoint = self._try_get_endpoint(name, cfg.namespace)
                    cfg.reuse_existing = True
                    continue
                if "Compute instance not available" in message:
                    raise
                raise

            except ConnectionError as exc:
                logger.warning("Connection error while waiting for endpoint: {}", exc)
                time.sleep(5)

        raise RuntimeError(f"Endpoint '{name}' 未能在 {MAX_TIME_FOR_SPINUP}s 内就绪")

    def _try_get_endpoint(self, name: str, namespace: Optional[str]):
        try:
            return self._hf["get_inference_endpoint"](name=name, namespace=namespace, token=self._token)
        except self._hf["HfHubHTTPError"]:
            return None

    @staticmethod
    def get_larger_hardware_suggestion(cur_instance_type: str = None, cur_instance_size: str = None):
        try:
            cur_idx = SORTED_INSTANCE_SIZES.index((cur_instance_type, cur_instance_size))
        except ValueError:
            cur_idx = -1
        next_idx = cur_idx + 1
        if next_idx >= len(SORTED_INSTANCE_SIZES):
            raise Exception(
                "To avoid accidental costs, we do not upgrade the current endpoint above 4 a100 automatically, please request it explicitly."
            )
        return SORTED_INSTANCE_SIZES[next_idx]

    @staticmethod
    def get_suggested_model_config(model_repo):
        url = f"https://ui.endpoints.huggingface.co/api/configuration?model_id={model_repo}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        config = response.json()

        suggested_compute = config.get("suggestedCompute")
        if not suggested_compute:
            raise ValueError("suggestedCompute missing from endpoint configuration response")
        suggested_vendor = suggested_compute.split("-")[0]
        suggested_region = suggested_compute.split("-")[1] if suggested_vendor == "azure" else "-".join(
            suggested_compute.split("-")[1:4]
        )
        suggested_instance = "-".join(suggested_compute.split("-")[-3:-1])
        suggested_size = suggested_compute.split("-")[-1]
        return suggested_vendor, suggested_region, suggested_instance, suggested_size

    def _get_dtype_args(self) -> Dict[str, str]:
        if self.config.dtype is None:
            return {}
        model_dtype = self.config.dtype.lower()
        quant_map = {
            "awq": "awq",
            "eetq": "eetq",
            "gptq": "gptq",
            "8bit": "bitsandbytes",
            "4bit": "bitsandbytes-nf4",
        }
        if model_dtype in quant_map:
            return {"QUANTIZE": quant_map[model_dtype]}
        if model_dtype in ["bfloat16", "float16"]:
            return {"DTYPE": model_dtype, "HF_INFERENCE_PRECISION": model_dtype}
        return {}

    def _get_custom_env_vars(self) -> Dict[str, str]:
        return {k: str(v) for k, v in self.config.env_vars.items()} if self.config.env_vars else {}

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = inputs.get("prompt") or ""
        stop_sequences = _extract_stop_sequences(inputs)
        sampling_params = inputs.get("sampling_params") or {}
        params = self._compose_generation_kwargs(sampling_params, stop_sequences)
        response = self._client.text_generation(prompt=prompt, wait_for_model=True, **params)
        return {"answer": _extract_hf_text(response), "raw_response": response}

    async def ainvoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._async_client is None or not self._enable_async:
            return await super().ainvoke(payload)

        inputs = self.prepare_inputs(payload)
        prompt = inputs.get("prompt") or ""
        stop_sequences = _extract_stop_sequences(inputs)
        sampling_params = inputs.get("sampling_params") or {}
        params = self._compose_generation_kwargs(sampling_params, stop_sequences)
        start = time.time()

        async def _do_call():
            return await self._async_client.text_generation(prompt=prompt, wait_for_model=True, **params)

        if self._async_semaphore:
            async with self._async_semaphore:
                response = await _do_call()
        else:
            response = await _do_call()
        return {"answer": _extract_hf_text(response), "raw_response": response, "latency_ms": (time.time() - start) * 1000}

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
