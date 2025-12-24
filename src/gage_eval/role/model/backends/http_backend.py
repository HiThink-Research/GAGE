"""HTTP-based backend (e.g., TGI, custom gateways)."""

from __future__ import annotations

import requests
from threading import Lock
from typing import Any, Dict

from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.registry import registry


@registry.asset(
    "backends",
    "http",
    desc="Generic HTTP text-generation backend",
    tags=("llm", "remote"),
    modalities=("text",),
)
class HTTPGenerationBackend(EngineBackend):
    """Backend that proxies prompts to an HTTP generation endpoint."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._session = requests.Session()
        self._lock = Lock()
        super().__init__(config)

    def load_model(self, config: Dict[str, Any]):
        base_url = config.get("base_url")
        if not base_url:
            raise ValueError("HTTPGenerationBackend requires 'base_url'")
        self.base_url = base_url.rstrip("/")
        self.endpoint = config.get("endpoint", "/generate")
        self.timeout = config.get("timeout", 60)
        self.headers = config.get("headers", {})
        self.request_fields = config.get("request_fields", {})
        self.response_path = config.get("response_path")

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._build_request(inputs)
        url = f"{self.base_url}{self.endpoint}"
        with self._lock:
            response = self._session.post(url, json=payload, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        text = self._extract_text(data)
        return {"answer": text, "raw_response": data}

    def close(self) -> None:
        self._session.close()

    def _build_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = inputs.get("prompt") or ""
        params = dict(inputs.get("sampling_params", {}))
        parameters = {
            "max_new_tokens": params.get("max_new_tokens", 512),
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "repetition_penalty": params.get("repetition_penalty"),
            "stop_sequences": params.get("stop"),
        }
        parameters = {k: v for k, v in parameters.items() if v is not None}
        body = {
            "inputs": prompt,
            "parameters": parameters,
        }
        body.update(self.request_fields)
        return body

    def _extract_text(self, data: Any) -> str:
        if isinstance(data, dict):
            if "generated_text" in data:
                return data["generated_text"]
            if "results" in data and data["results"]:
                return data["results"][0].get("text", "")
            if self.response_path:
                return _dig(data, self.response_path.split("."))
        if isinstance(data, list) and data:
            if isinstance(data[0], dict):
                return data[0].get("generated_text") or data[0].get("text", "")
            if isinstance(data[0], str):
                return data[0]
        return str(data)


def _dig(node: Dict[str, Any], path):
    current = node
    for part in path:
        if not isinstance(current, dict):
            return ""
        current = current.get(part)
        if current is None:
            return ""
    return current if isinstance(current, str) else str(current)
