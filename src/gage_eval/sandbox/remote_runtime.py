"""Remote sandbox runtime implementation (API-backed)."""

from __future__ import annotations

import base64
import shlex
import time
from typing import Any, Dict, Optional

import requests

from gage_eval.sandbox.base import BaseSandbox, ExecResult, SandboxOptionalMixin


class RemoteSandbox(SandboxOptionalMixin, BaseSandbox):
    """Remote sandbox that proxies exec requests to a remote platform."""

    def __init__(
        self,
        runtime_configs: Optional[Dict[str, Any]] = None,
        resources: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._runtime_configs = dict(runtime_configs or {})
        self._resources = dict(resources or {})
        self._running = False
        self._sandbox_id: Optional[str] = None
        self._control_endpoint = self._runtime_configs.get("control_endpoint")
        self._data_endpoint = self._runtime_configs.get("data_endpoint")
        self._exec_url = self._runtime_configs.get("exec_url")
        self._exec_path = self._runtime_configs.get("exec_path", "run_command")
        self._file_read_url = self._runtime_configs.get("file_read_url")
        self._file_write_url = self._runtime_configs.get("file_write_url")
        self._file_read_path = self._runtime_configs.get("file_read_path", "read_file")
        self._file_write_path = self._runtime_configs.get(
            "file_write_path", "write_file"
        )
        self._env_endpoint = self._runtime_configs.get(
            "env_endpoint"
        ) or self._runtime_configs.get("environment_endpoint")
        self._apis_endpoint = self._runtime_configs.get("apis_endpoint")
        self._mcp_endpoint = self._runtime_configs.get("mcp_endpoint")
        self._timeout_s = int(self._runtime_configs.get("timeout_s", 30))
        self._file_timeout_s = int(
            self._runtime_configs.get("file_timeout_s", self._timeout_s)
        )
        self._startup_timeout_s = int(
            self._runtime_configs.get("startup_timeout_s", 120)
        )
        self._max_retries = int(self._runtime_configs.get("max_retries", 3))
        self._retry_backoff = float(
            self._runtime_configs.get("retry_backoff_factor", 1.5)
        )
        self._headers = dict(self._runtime_configs.get("headers") or {})
        self._requester = self._runtime_configs.get("requester")
        self._auth_type = self._runtime_configs.get("auth_type")
        self._auth_token = self._runtime_configs.get("auth_token")
        self._api_key_header = str(
            self._runtime_configs.get("api_key_header", "X-API-Key")
        )

    def start(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Start the remote sandbox session.

        Args:
            config: Sandbox configuration payload.

        Returns:
            Runtime handle metadata for downstream consumers.
        """

        self._merge_start_config(config)

        # Create managed sandbox when a control plane exists.
        platform_response: Dict[str, Any] = {}
        if self._control_endpoint:
            create_payload = {
                "image": config.get("image"),
                "resources": config.get("resources") or self._resources or {},
                "env": (config.get("runtime_configs") or {}).get("env", {}),
            }
            platform_response = self._platform_request(
                f"{str(self._control_endpoint).rstrip('/')}/sandboxes",
                create_payload,
                method="POST",
                timeout_s=self._startup_timeout_s,
            )
            sandbox_id = platform_response.get("sandbox_id") or platform_response.get(
                "id"
            )
            if sandbox_id is not None:
                self._sandbox_id = str(sandbox_id)
            self._apply_platform_response(platform_response)
            if self._sandbox_id:
                ready_response = self._wait_for_ready(timeout_s=self._startup_timeout_s)
                if isinstance(ready_response, dict):
                    self._apply_platform_response(ready_response)

        # Mark running and expose the runtime handle.
        self._running = True
        return self._build_runtime_handle()

    def _merge_start_config(self, config: Dict[str, Any]) -> None:
        """Merge start-time config overrides into instance attributes."""

        self._runtime_configs.update(config.get("runtime_configs", {}) or {})
        rc = self._runtime_configs

        self._control_endpoint = _first_non_empty(
            config.get("control_endpoint"), rc.get("control_endpoint"),
            self._control_endpoint,
        )
        self._data_endpoint = _first_non_empty(
            config.get("data_endpoint"), rc.get("data_endpoint"),
            self._data_endpoint,
        )
        self._exec_url = _first_non_empty(
            config.get("exec_url"), rc.get("exec_url"), self._exec_url,
        )
        self._exec_path = str(rc.get("exec_path", self._exec_path))
        self._file_read_url = _first_non_empty(
            config.get("file_read_url"), rc.get("file_read_url"),
            self._file_read_url,
        )
        self._file_write_url = _first_non_empty(
            config.get("file_write_url"), rc.get("file_write_url"),
            self._file_write_url,
        )
        self._file_read_path = str(rc.get("file_read_path", self._file_read_path))
        self._file_write_path = str(rc.get("file_write_path", self._file_write_path))
        self._env_endpoint = _first_non_empty(
            config.get("env_endpoint"), config.get("environment_endpoint"),
            rc.get("env_endpoint"), rc.get("environment_endpoint"),
            self._env_endpoint,
        )
        self._apis_endpoint = _first_non_empty(
            config.get("apis_endpoint"), rc.get("apis_endpoint"),
            self._apis_endpoint,
        )
        self._mcp_endpoint = _first_non_empty(
            config.get("mcp_endpoint"), rc.get("mcp_endpoint"),
            self._mcp_endpoint,
        )
        self._timeout_s = int(rc.get("timeout_s", self._timeout_s))
        self._file_timeout_s = int(rc.get("file_timeout_s", self._file_timeout_s))
        self._startup_timeout_s = int(
            rc.get("startup_timeout_s", self._startup_timeout_s)
        )
        self._max_retries = int(rc.get("max_retries", self._max_retries))
        self._retry_backoff = float(
            rc.get("retry_backoff_factor", self._retry_backoff)
        )
        self._headers = dict(rc.get("headers") or self._headers)
        self._requester = rc.get("requester") or self._requester
        self._auth_type = _first_non_empty(
            config.get("auth_type"), rc.get("auth_type"), self._auth_type,
        )
        self._auth_token = _first_non_empty(
            config.get("auth_token"), rc.get("auth_token"), self._auth_token,
        )
        self._api_key_header = str(rc.get("api_key_header", self._api_key_header))

    def exec(
        self,
        command: str,
        timeout: int | None = None,
        *,
        login_shell: bool = True,
    ) -> ExecResult:
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
        result.duration_ms = (
            result.duration_ms or (time.perf_counter() - start) * 1000.0
        )
        return result

    def read_file(self, path: str) -> bytes:
        """Read a file from the remote sandbox."""

        read_url = self._resolve_file_url(self._file_read_url, self._file_read_path)
        if read_url:
            payload = dict(self._runtime_configs.get("file_read_payload") or {})
            payload["path"] = path
            response = self._request(read_url, payload, timeout_s=self._file_timeout_s)
            return _extract_file_bytes(response)
        result = self.exec(f"cat {shlex.quote(path)}", timeout=self._file_timeout_s)
        if result.exit_code != 0:
            raise RuntimeError("remote_read_failed")
        return result.stdout.encode("utf-8")

    def write_file(self, path: str, content: bytes) -> None:
        """Write a file into the remote sandbox."""

        if isinstance(content, str):
            content = content.encode("utf-8")
        write_url = self._resolve_file_url(self._file_write_url, self._file_write_path)
        if write_url:
            payload = dict(self._runtime_configs.get("file_write_payload") or {})
            payload["path"] = path
            payload["content_b64"] = base64.b64encode(content).decode("ascii")
            payload.setdefault("encoding", "base64")
            self._request(write_url, payload, timeout_s=self._file_timeout_s)
            return
        command = _build_write_command(path, content)
        result = self.exec(command, timeout=self._file_timeout_s)
        if result.exit_code != 0:
            raise RuntimeError("remote_write_failed")

    def teardown(self) -> None:
        """Tear down the managed remote sandbox session."""

        if self._control_endpoint and self._sandbox_id:
            try:
                self._platform_request(
                    f"{str(self._control_endpoint).rstrip('/')}/sandboxes/{self._sandbox_id}",
                    {},
                    method="DELETE",
                    timeout_s=30,
                )
            except Exception:
                pass
        self._running = False
        self._sandbox_id = None

    def is_alive(self, timeout_s: float | None = None) -> bool:
        """Return whether the managed sandbox is still healthy."""

        if not self._running:
            return False
        if self._control_endpoint and self._sandbox_id:
            try:
                response = self._platform_request(
                    f"{str(self._control_endpoint).rstrip('/')}/sandboxes/{self._sandbox_id}",
                    {},
                    method="GET",
                    timeout_s=int(timeout_s or 5),
                )
                status = str(response.get("status") or "").strip().lower()
                self._apply_platform_response(response)
                return status in {"running", "ready", "started", "active"}
            except Exception:
                return False
        exec_url = self._resolve_exec_url()
        if not exec_url:
            return self._running
        try:
            result = self.exec("echo 1", timeout=int(timeout_s or 5))
            return result.exit_code == 0
        except Exception:
            return False

    def _build_runtime_handle(self) -> Dict[str, Any]:
        handle: Dict[str, Any] = {
            "control_endpoint": self._control_endpoint,
            "data_endpoint": self._data_endpoint,
            "exec_url": self._resolve_exec_url(),
            "sandbox_id": self._sandbox_id,
        }
        optional_values = {
            "env_endpoint": self._env_endpoint,
            "apis_endpoint": self._apis_endpoint,
            "mcp_endpoint": self._mcp_endpoint,
            "file_read_url": self._file_read_url,
            "file_write_url": self._file_write_url,
        }
        for key, value in optional_values.items():
            if value:
                handle[key] = value
        return {key: value for key, value in handle.items() if value is not None}

    def _apply_platform_response(self, response: Dict[str, Any]) -> None:
        if not isinstance(response, dict):
            return
        self._data_endpoint = response.get("data_endpoint") or self._data_endpoint
        self._exec_url = response.get("exec_url") or self._exec_url
        self._file_read_url = response.get("file_read_url") or self._file_read_url
        self._file_write_url = response.get("file_write_url") or self._file_write_url
        self._env_endpoint = (
            response.get("env_endpoint")
            or response.get("environment_endpoint")
            or self._env_endpoint
        )
        self._apis_endpoint = response.get("apis_endpoint") or self._apis_endpoint
        self._mcp_endpoint = response.get("mcp_endpoint") or self._mcp_endpoint

    def _resolve_exec_url(self) -> Optional[str]:
        if self._exec_url:
            return self._exec_url
        if not self._data_endpoint:
            return None
        base = str(self._data_endpoint).rstrip("/")
        path = str(self._exec_path).lstrip("/")
        return f"{base}/{path}"

    def _resolve_file_url(self, explicit: Optional[str], path: str) -> Optional[str]:
        if explicit:
            return explicit
        if not self._data_endpoint:
            return None
        base = str(self._data_endpoint).rstrip("/")
        suffix = str(path).lstrip("/")
        return f"{base}/{suffix}"

    def _request(self, url: str, payload: Dict[str, Any], *, timeout_s: int) -> Any:
        headers = self._build_auth_headers()
        if callable(self._requester):
            return _call_requester(self._requester, url, payload, timeout_s, headers)
        response = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
        response.raise_for_status()
        return _decode_response_payload(response)

    def _platform_request(
        self,
        url: str,
        payload: Dict[str, Any],
        *,
        method: str = "POST",
        timeout_s: Optional[int] = None,
    ) -> Any:
        headers = self._build_auth_headers()
        request_timeout_s = int(timeout_s or self._startup_timeout_s)
        normalized_method = str(method or "POST").upper()
        last_error: Optional[BaseException] = None
        for attempt in range(self._max_retries + 1):
            try:
                if callable(self._requester):
                    return _call_requester(
                        self._requester,
                        url,
                        payload,
                        request_timeout_s,
                        headers,
                        method=normalized_method,
                    )
                if normalized_method == "GET":
                    response = requests.get(
                        url, headers=headers, timeout=request_timeout_s
                    )
                elif normalized_method == "DELETE":
                    response = requests.delete(
                        url, headers=headers, timeout=request_timeout_s
                    )
                else:
                    response = requests.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=request_timeout_s,
                    )
                response.raise_for_status()
                return _decode_response_payload(response)
            except requests.exceptions.RequestException as exc:
                last_error = exc
            except Exception as exc:  # pragma: no cover - requester-specific path
                last_error = exc
            if attempt >= self._max_retries:
                break
            time.sleep(max(0.0, self._retry_backoff ** max(0, attempt - 1)))
        if last_error is not None:
            raise last_error
        raise RuntimeError("platform_request_exhausted")

    def _wait_for_ready(self, timeout_s: int = 120) -> Dict[str, Any]:
        """Poll the platform until the remote sandbox becomes ready."""

        if not self._control_endpoint or not self._sandbox_id:
            return {}
        deadline = time.monotonic() + max(1, int(timeout_s))
        status_url = (
            f"{str(self._control_endpoint).rstrip('/')}/sandboxes/{self._sandbox_id}"
        )
        last_response: Dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                response = self._platform_request(
                    status_url,
                    {},
                    method="GET",
                    timeout_s=min(self._startup_timeout_s, 30),
                )
            except Exception:
                time.sleep(2)
                continue
            if isinstance(response, dict):
                last_response = response
                self._apply_platform_response(response)
                status = str(response.get("status") or "").strip().lower()
                if status in {"running", "ready", "started", "active"}:
                    return response
                if status in {"failed", "error", "terminated"}:
                    raise RuntimeError(f"sandbox_creation_failed: status={status}")
            time.sleep(2)
        raise TimeoutError(f"sandbox not ready after {timeout_s}s")

    def _build_auth_headers(self) -> Dict[str, str]:
        headers = dict(self._headers)
        if self._auth_type == "bearer" and self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        elif self._auth_type == "api_key" and self._auth_token:
            headers[self._api_key_header] = str(self._auth_token)
        return headers


def _call_requester(
    requester: Any,
    url: str,
    payload: Dict[str, Any],
    timeout_s: int,
    headers: Dict[str, str],
    *,
    method: Optional[str] = None,
) -> Any:
    if method is not None:
        try:
            return requester(url, payload, timeout_s, headers, method=method)
        except TypeError:
            pass
    return requester(url, payload, timeout_s, headers)


def _decode_response_payload(response: requests.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return {"stdout": response.text, "stderr": "", "exit_code": 0}


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


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
            exit_code=int(
                payload.get("exit_code")
                or payload.get("returncode")
                or payload.get("code")
                or 0
            ),
            stdout=str(payload.get("stdout") or payload.get("output") or ""),
            stderr=str(payload.get("stderr") or payload.get("error") or ""),
            duration_ms=float(
                payload.get("duration_ms") or payload.get("latency_ms") or 0.0
            ),
        )
    return ExecResult(exit_code=0, stdout=str(raw), stderr="", duration_ms=0.0)


def _extract_file_bytes(raw: Any) -> bytes:
    payload = raw
    if isinstance(payload, dict):
        if isinstance(payload.get("result"), dict):
            payload = payload["result"]
        if isinstance(payload.get("data"), dict):
            payload = payload["data"]
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    if isinstance(payload, str):
        return payload.encode("utf-8")
    if isinstance(payload, dict):
        if "content_b64" in payload:
            return base64.b64decode(payload.get("content_b64") or "")
        if "content" in payload:
            content = payload.get("content")
            if isinstance(content, (bytes, bytearray)):
                return bytes(content)
            if isinstance(content, str):
                return content.encode("utf-8")
    return str(payload).encode("utf-8")


def _build_write_command(path: str, content: bytes) -> str:
    b64 = base64.b64encode(content).decode("ascii")
    return (
        "python - <<'PY'\n"
        "import base64\n"
        "from pathlib import Path\n"
        f"path = {path!r}\n"
        f"data = base64.b64decode({b64!r})\n"
        "target = Path(path)\n"
        "target.parent.mkdir(parents=True, exist_ok=True)\n"
        "target.write_bytes(data)\n"
        "PY\n"
    )
