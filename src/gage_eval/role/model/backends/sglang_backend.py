"""SGLang backend wrapper."""

from __future__ import annotations

import subprocess
import time
from typing import Any, Dict, Optional

import requests

from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.registry import registry


@registry.asset(
    "backends",
    "sglang",
    desc="SGLang 推理服务后端",
    tags=("llm", "remote"),
    modalities=("text",),
)
class SGLangBackend(EngineBackend):
    """Launches (optionally) and talks to an SGLang server via HTTP."""

    def load_model(self, config: Dict[str, Any]):
        self.base_url = config.get("base_url")
        if not self.base_url:
            host = config.get("host", "127.0.0.1")
            port = config.get("port", 30000)
            self.base_url = f"http://{host}:{port}"
        self.timeout = config.get("timeout", 120)
        self.session = requests.Session()
        self._process: Optional[subprocess.Popen] = None
        self.default_sampling = {
            "max_new_tokens": config.get("max_new_tokens", 512),
            "temperature": config.get("temperature"),
            "top_p": config.get("top_p"),
            "repetition_penalty": config.get("repetition_penalty"),
            "presence_penalty": config.get("presence_penalty"),
            "stop": config.get("stop"),
        }

        if command := config.get("launch_command"):
            env = dict(config.get("launch_env", {})) or None
            self._process = subprocess.Popen(command, shell=True, env=env)
            self._wait_for_server(config.get("startup_timeout", 600))

    def _wait_for_server(self, timeout: int) -> None:
        end = time.time() + timeout
        url = f"{self.base_url}/v1/models"
        while time.time() < end:
            try:
                r = self.session.get(url, timeout=5)
                if r.status_code == 200:
                    time.sleep(2)
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError("Timed out waiting for SGLang server to become ready")

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_body = self._build_request(payload)
        response = self.session.post(
            f"{self.base_url}/generate",
            json=request_body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        output_type = payload.get("output_type") or payload.get("sample", {}).get("output_type")
        if output_type == "next_token_prob":
            return {"top_logprobs": _extract_logprobs(data)}
        text = _extract_text(data)
        return {"answer": text, "raw_response": data}

    def _build_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sampling_params = dict(self.default_sampling)
        sampling_params.update(payload.get("sampling_params") or {})
        request = {
            "sampling_params": {
                k: v
                for k, v in sampling_params.items()
                if v is not None and k != "stop"
            }
        }
        if isinstance(payload.get("inputs"), list):
            request["input_ids"] = payload["inputs"]
        else:
            text = payload.get("prompt") or payload.get("text") or payload.get("sample", {}).get("prompt", "")
            request["text"] = text  # SGLang >=0.5 expects `text`
            request["prompt"] = text  # backward compat with older servers
        if sampling_params.get("stop"):
            request["stop"] = sampling_params["stop"]
        if payload.get("logprob_token_ids"):
            request["logprob_token_ids"] = payload["logprob_token_ids"]
        return request

    def shutdown(self) -> None:  # pragma: no cover - subprocess housekeeping
        self.session.close()
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None


def _extract_text(data: Any) -> str:
    if isinstance(data, dict):
        if "text" in data and isinstance(data["text"], str):
            return data["text"]
        if "text" in data and isinstance(data["text"], list):
            return "".join(item.get("text", "") for item in data["text"])
        if "outputs" in data and data["outputs"]:
            return data["outputs"][0].get("text", "")
        if "meta_info" in data and "output_text" in data["meta_info"]:
            return data["meta_info"]["output_text"]
    if isinstance(data, list) and data and isinstance(data[0], str):
        return data[0]
    return str(data)


def _extract_logprobs(data: Dict[str, Any]):
    try:
        entries = data["meta_info"]["output_top_logprobs"][0]
    except (KeyError, IndexError, TypeError):
        return []
    return [[token_id, logprob] for logprob, token_id, _ in entries]
