"""Text Generation Inference backend."""

from __future__ import annotations

import subprocess
import time
from typing import Any, Dict, Optional

import requests

from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.registry import registry


@registry.asset(
    "backends",
    "tgi",
    desc="HuggingFace TGI 服务推理后端",
    tags=("llm", "remote"),
    modalities=("text",),
)
class TGIBackend(EngineBackend):
    """Minimal wrapper for HuggingFace TGI servers."""

    def load_model(self, config: Dict[str, Any]):
        self.base_url = config.get("base_url")
        if not self.base_url:
            host = config.get("host", "127.0.0.1")
            port = config.get("port", 8080)
            self.base_url = f"http://{host}:{port}"
        self.timeout = config.get("timeout", 120)
        self.session = requests.Session()
        self.default_parameters = {
            "max_new_tokens": config.get("max_new_tokens", 512),
            "temperature": config.get("temperature"),
            "top_p": config.get("top_p"),
            "repetition_penalty": config.get("repetition_penalty"),
            "stop_sequences": config.get("stop"),
            "return_full_text": False,
        }
        self._process: Optional[subprocess.Popen] = None
        if command := config.get("launch_command"):
            env = dict(config.get("launch_env", {})) or None
            self._process = subprocess.Popen(command, shell=True, env=env)
            self._wait_for_server(config.get("startup_timeout", 600))

    def _wait_for_server(self, timeout: int) -> None:
        end = time.time() + timeout
        url = f"{self.base_url}/health"
        while time.time() < end:
            try:
                r = self.session.get(url, timeout=5)
                if r.status_code == 200:
                    time.sleep(2)
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError("Timed out waiting for TGI server to become ready")

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        parameters = dict(self.default_parameters)
        parameters.update(payload.get("sampling_params") or {})
        body = {
            "inputs": payload.get("prompt") or payload.get("sample", {}).get("prompt", ""),
            "parameters": {k: v for k, v in parameters.items() if v is not None},
        }
        response = self.session.post(
            f"{self.base_url}/generate",
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        text = _extract_tgi_text(data)
        return {"answer": text, "raw_response": data}

    def shutdown(self) -> None:  # pragma: no cover
        self.session.close()
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None


def _extract_tgi_text(data: Any) -> str:
    if isinstance(data, dict):
        if "generated_text" in data:
            return data["generated_text"]
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("text", "")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return first.get("generated_text") or first.get("text", "")
        if isinstance(first, str):
            return first
    return str(data)
