"""Remote sandbox runtime implementation (API-backed)."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests

from gage_eval.sandbox.base import BaseSandbox, ExecResult, SandboxOptionalMixin


class RemoteSandbox(SandboxOptionalMixin, BaseSandbox):
    """Remote sandbox that proxies exec requests to a control endpoint."""

    def __init__(self, runtime_configs: Optional[Dict[str, Any]] = None, resources: Optional[Dict[str, Any]] = None) -> None:
        self._runtime_configs = dict(runtime_configs or {})
        self._resources = dict(resources or {})
        self._running = False
        self._control_endpoint = self._runtime_configs.get("control_endpoint")
        self._data_endpoint = self._runtime_configs.get("data_endpoint")
        self._exec_url = self._runtime_configs.get("exec_url")
        self._exec_path = self._runtime_configs.get("exec_path", "run_command")
        self._timeout_s = int(self._runtime_configs.get("timeout_s", 30))
        self._headers = dict(self._runtime_configs.get("headers") or {})
        self._requester = self._runtime_configs.get("requester")

    def start(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Start the remote sandbox session.

        Args:
            config: Sandbox configuration payload.

        Returns:
            Runtime handle metadata for downstream consumers.
        """

        self._runtime_configs.update(config.get("runtime_configs", {}) or {})
        self._control_endpoint = config.get("control_endpoint") or self._control_endpoint
        self._data_endpoint = config.get("data_endpoint") or self._data_endpoint
        self._exec_url = config.get("exec_url") or self._exec_url
        self._exec_path = self._runtime_configs.get("exec_path", self._exec_path)
        self._timeout_s = int(self._runtime_configs.get("timeout_s", self._timeout_s))
        self._headers = dict(self._runtime_configs.get("headers") or self._headers)
        self._requester = self._runtime_configs.get("requester") or self._requester
        self._running = True
        return {
            "control_endpoint": self._control_endpoint,
            "data_endpoint": self._data_endpoint,
            "exec_url": self._resolve_exec_url(),
        }

    def exec(self, command: str, timeout: int | None = None) -> ExecResult:
        """Execute a command via the remote execution endpoint.

        Args:
            command: Command payload to execute remotely.
            timeout: Per-call timeout in seconds.

        Returns:
            Normalized execution result payload.
        """

        runner = self._runtime_configs.get("command_runner")
        if callable(runner):
            return _normalize_exec_result(runner(command, timeout))
        start = time.perf_counter()
        exec_url = self._resolve_exec_url()
        if not exec_url:
            raise RuntimeError("remote_exec_url_missing")
        payload = dict(self._runtime_configs.get("exec_payload") or {})
        payload["command"] = command
        timeout_s = int(self._timeout_s if timeout is None else timeout)
        response = self._request(exec_url, payload, timeout_s=timeout_s)
        result = _normalize_exec_result(response)
        result.duration_ms = result.duration_ms or (time.perf_counter() - start) * 1000.0
        return result

    def teardown(self) -> None:
        self._running = False

    def is_alive(self, timeout_s: float | None = None) -> bool:
        return self._running

    def _resolve_exec_url(self) -> Optional[str]:
        if self._exec_url:
            return self._exec_url
        if not self._data_endpoint:
            return None
        base = str(self._data_endpoint).rstrip("/")
        path = str(self._exec_path).lstrip("/")
        return f"{base}/{path}"

    def _request(self, url: str, payload: Dict[str, Any], *, timeout_s: int) -> Any:
        if callable(self._requester):
            return self._requester(url, payload, timeout_s, dict(self._headers))
        response = requests.post(url, json=payload, headers=self._headers, timeout=timeout_s)
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return {"stdout": response.text, "stderr": "", "exit_code": 0}


def _normalize_exec_result(raw: Any) -> ExecResult:
    if isinstance(raw, ExecResult):
        return raw
    if isinstance(raw, dict):
        payload = raw
        if isinstance(payload.get("result"), dict):
            payload = payload["result"]
        if isinstance(payload.get("data"), dict):
            payload = payload["data"]
        return ExecResult(
            exit_code=int(payload.get("exit_code") or payload.get("returncode") or payload.get("code") or 0),
            stdout=str(payload.get("stdout") or payload.get("output") or ""),
            stderr=str(payload.get("stderr") or payload.get("error") or ""),
            duration_ms=float(payload.get("duration_ms") or payload.get("latency_ms") or 0.0),
        )
    return ExecResult(exit_code=0, stdout=str(raw), stderr="", duration_ms=0.0)
