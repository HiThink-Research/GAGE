"""Vendor-specific HTTP backends (Claude / Gemini / OpenAI Batch)."""

from __future__ import annotations

import base64
import io
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from loguru import logger
from PIL import Image

from gage_eval.registry import registry
from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.role.model.config.vendor_http import (
    ClaudeBackendConfig,
    GeminiBackendConfig,
    OpenAIBatchBackendConfig,
)


# ---------------------------------------------------------------------------
# Claude HTTP Backend
# ---------------------------------------------------------------------------
@registry.asset(
    "backends",
    "claude_http",
    desc="Anthropic Claude 多模态 HTTP 后端",
    tags=("llm", "remote", "anthropic"),
    modalities=("text", "vision"),
    config_schema_ref="gage_eval.role.model.config.vendor_http:ClaudeBackendConfig",
)
class ClaudeHTTPBackend(EngineBackend):
    def load_model(self, config: Dict[str, Any]):
        try:  # pragma: no cover
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("claude_http backend requires the 'anthropic' package") from exc

        self._cfg = ClaudeBackendConfig(**config)
        api_key = self._cfg.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ClaudeHTTPBackend requires api_key or ANTHROPIC_API_KEY env")
        self._client = anthropic.Anthropic(api_key=api_key, base_url=self._cfg.base_url)
        return self._client

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        sample = inputs.get("sample", {})
        system_prompt = inputs.get("system_prompt") or sample.get("system_prompt") or self._cfg.default_system
        content = self._build_content(inputs)
        sampling = inputs.get("sampling_params") or {}
        max_tokens = sampling.get("max_new_tokens") or self._cfg.max_output_tokens
        temperature = sampling.get("temperature", self._cfg.temperature)
        top_p = sampling.get("top_p", self._cfg.top_p)
        response = self._client.messages.create(
            model=self._cfg.model,
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[{"role": "user", "content": content}],
        )
        text = _anthropic_text(response)
        return {"answer": text, "raw_response": response.model_dump()}

    def _build_content(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        prompt = inputs.get("prompt") or ""
        if prompt:
            content.append({"type": "text", "text": prompt})
        images = _extract_image_b64(inputs)
        for data in images:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": data,
                    },
                }
            )
        return content or [{"type": "text", "text": ""}]


# ---------------------------------------------------------------------------
# Gemini HTTP Backend
# ---------------------------------------------------------------------------
@registry.asset(
    "backends",
    "gemini_http",
    desc="Google Gemini 多模态 HTTP 后端",
    tags=("llm", "remote", "gemini"),
    modalities=("text", "vision", "audio"),
    config_schema_ref="gage_eval.role.model.config.vendor_http:GeminiBackendConfig",
)
class GeminiHTTPBackend(EngineBackend):
    def load_model(self, config: Dict[str, Any]):
        try:  # pragma: no cover
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("gemini_http backend requires google-generativeai") from exc

        self._cfg = GeminiBackendConfig(**config)
        api_key = self._cfg.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GeminiHTTPBackend requires api_key or GOOGLE_API_KEY env")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(self._cfg.model)
        return self._model

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        contents = self._build_contents(inputs)
        sampling = inputs.get("sampling_params") or {}
        params = self._cfg.generation_parameters.to_dict()
        if sampling:
            params.update({k: v for k, v in sampling.items() if v is not None})
        generation_config = {"max_output_tokens": params.get("max_new_tokens", 512)}
        if params.get("temperature") is not None:
            generation_config["temperature"] = params["temperature"]
        if params.get("top_p") is not None:
            generation_config["top_p"] = params["top_p"]
        if params.get("top_k") is not None:
            generation_config["top_k"] = params["top_k"]
        try:
            response = self._model.generate_content(
                contents,
                generation_config=generation_config,
                safety_settings=self._cfg.safety_settings or None,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Gemini request failed: {exc}") from exc
        text = response.text if hasattr(response, "text") else str(response)
        return {"answer": text, "raw_response": getattr(response, "_result", None)}

    def _build_contents(self, inputs: Dict[str, Any]) -> List[Any]:
        contents: List[Any] = []
        prompt = inputs.get("prompt") or ""
        if prompt:
            contents.append(prompt)
        images = _extract_image_objects(inputs)
        contents.extend(images)
        return contents or [""]


# ---------------------------------------------------------------------------
# OpenAI Batch Backend
# ---------------------------------------------------------------------------
@registry.asset(
    "backends",
    "openai_batch",
    desc="OpenAI Batch Chat Completion 后端",
    tags=("llm", "remote", "openai"),
    modalities=("text",),
    config_schema_ref="gage_eval.role.model.config.vendor_http:OpenAIBatchBackendConfig",
)
class OpenAIBatchBackend(EngineBackend):
    def load_model(self, config: Dict[str, Any]):
        try:  # pragma: no cover
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai_batch backend requires the 'openai' package") from exc

        self._cfg = OpenAIBatchBackendConfig(**config)
        api_key = self._cfg.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAIBatchBackend requires api_key or OPENAI_API_KEY env")
        client_args = dict(self._cfg.extra_client_args or {})
        if self._cfg.base_url:
            client_args["base_url"] = self._cfg.base_url
        self._client = OpenAI(api_key=api_key, **client_args)
        return self._client

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        request_body = self._build_request(inputs)
        custom_id = f"gage-{uuid.uuid4()}"
        json_line = json.dumps(
            {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": request_body}
        )
        input_bytes = io.BytesIO(json_line.encode("utf-8"))
        upload = self._client.files.create(file=("batch.jsonl", input_bytes), purpose="batch")
        batch = self._client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=self._cfg.metadata or None,
        )
        output = self._wait_batch(batch.id)
        if not output:
            raise RuntimeError("Batch completed without output file")
        content = self._client.files.content(output)
        text = self._parse_batch_output(content, custom_id)
        if not self._cfg.keep_files:
            self._cleanup(upload.id, output)
        return {"answer": text, "raw_response": content}

    def _wait_batch(self, batch_id: str) -> Optional[str]:
        deadline = time.time() + self._cfg.timeout
        while time.time() < deadline:
            info = self._client.batches.retrieve(batch_id)
            status = info.status
            if status == "completed":
                return info.output_file_id
            if status == "failed":
                raise RuntimeError(f"OpenAI batch failed: {info}")
            time.sleep(self._cfg.poll_interval)
        raise TimeoutError(f"Timed out waiting for batch {batch_id}")

    def _parse_batch_output(self, blob, custom_id: str) -> str:
        text = blob.text if hasattr(blob, "text") else str(blob)
        for line in text.splitlines():
            payload = json.loads(line)
            if payload.get("custom_id") == custom_id:
                body = payload.get("response", {}).get("body", {})
                choices = body.get("choices") or []
                if choices:
                    return choices[0]["message"].get("content") or ""
        return ""

    def _cleanup(self, input_file_id: str, output_file_id: str) -> None:
        try:
            self._client.files.delete(input_file_id)
        except Exception:
            pass
        try:
            self._client.files.delete(output_file_id)
        except Exception:
            pass

    def _build_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        sample = inputs.get("sample", {})
        messages = _compose_openai_messages(inputs)
        params = self._cfg.generation_parameters.to_dict()
        sampling = inputs.get("sampling_params") or {}
        params.update({k: v for k, v in sampling.items() if v is not None})
        request = {
            "model": inputs.get("model") or self._cfg.model,
            "messages": messages,
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "max_tokens": params.get("max_new_tokens"),
            "presence_penalty": params.get("presence_penalty"),
            "frequency_penalty": params.get("frequency_penalty"),
            "stop": params.get("stop"),
        }
        if sample.get("tools"):
            request["tools"] = sample["tools"]
        return {k: v for k, v in request.items() if v is not None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _extract_image_paths(inputs: Dict[str, Any]) -> List[str]:
    sample = inputs.get("sample", {})
    raw_inputs = inputs.get("inputs") or sample.get("inputs") or {}
    mm = raw_inputs.get("multi_modal_data") if isinstance(raw_inputs, dict) else None
    if not mm:
        mm = sample.get("multi_modal_data")
    paths: List[str] = []
    if isinstance(mm, dict):
        images = mm.get("image") or mm.get("images")
        if isinstance(images, list):
            for item in images:
                if isinstance(item, str):
                    paths.append(item)
        elif isinstance(images, str):
            paths.append(images)
    return paths


def _extract_image_b64(inputs: Dict[str, Any]) -> List[str]:
    paths = _extract_image_paths(inputs)
    results = []
    for path in paths:
        try:
            with Image.open(path) as img:
                results.append(_encode_image(img))
        except Exception as exc:
            logger.warning("Failed to load image %s: %s", path, exc)
    return results


def _extract_image_objects(inputs: Dict[str, Any]) -> List[Any]:
    paths = _extract_image_paths(inputs)
    images = []
    for path in paths:
        try:
            images.append(Image.open(path).convert("RGB"))
        except Exception as exc:
            logger.warning("Failed to load image %s: %s", path, exc)
    return images


def _compose_openai_messages(inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    sample = inputs.get("sample", {})
    messages = inputs.get("messages") or sample.get("messages")
    if messages:
        return messages
    system_prompt = inputs.get("system_prompt") or sample.get("system_prompt")
    prompt = inputs.get("prompt") or sample.get("prompt") or sample.get("text") or ""
    built: List[Dict[str, Any]] = []
    if system_prompt:
        built.append({"role": "system", "content": system_prompt})
    built.append({"role": "user", "content": prompt})
    return built


def _anthropic_text(response) -> str:
    parts = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
    return "".join(parts)
