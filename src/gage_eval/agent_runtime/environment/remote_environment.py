"""Remote environment wrapper."""

from __future__ import annotations

from pathlib import Path
import shlex
from typing import Any, Mapping, Optional, Type

from gage_eval.sandbox.contracts import (
    RemoteSandboxContract,
    coerce_remote_sandbox_handle,
    serialize_handle,
    validate_remote_sandbox_contract,
)
from gage_eval.sandbox.base import ExecResult
from gage_eval.sandbox.surfaces import ClientSurface, build_remote_surfaces


class RemoteEnvironment:
    """Remote-backed AgentEnvironment implementation."""

    def __init__(
        self,
        *,
        contract: Optional[RemoteSandboxContract] = None,
        config: Optional[dict[str, Any]] = None,
        runtime_configs: Optional[dict[str, Any]] = None,
        resources: Optional[dict[str, Any]] = None,
        sandbox: Any = None,
        sandbox_cls: Optional[Type[Any]] = None,
    ) -> None:
        self._contract = contract
        self._config = dict(config or {})
        self._runtime_configs = dict(runtime_configs or {})
        self._resources = dict(resources or {})
        self._sandbox = sandbox
        self._sandbox_cls = sandbox_cls
        self._started = False
        self._runtime_handle: dict[str, Any] = {}
        self._surfaces: dict[str, ClientSurface] = {}

    def start(self) -> dict:
        if self._started:
            return dict(self._runtime_handle)
        sandbox = self._ensure_sandbox()
        start_config = self._build_start_config()
        handle = sandbox.start(start_config)
        self._runtime_handle = dict(handle or {})
        if self._contract:
            self._runtime_handle.setdefault("mode", self._contract.mode)
            self._runtime_handle.setdefault("remote_mode", self._contract.mode)
            self._runtime_handle.setdefault(
                "workspace_root",
                self._contract.workspace_root or self._contract.attach_target,
            )
            self._runtime_handle.setdefault("provider", self._contract.provider)
            if self._contract.attach_target:
                self._runtime_handle.setdefault("attach_target", self._contract.attach_target)
            normalized_handle = coerce_remote_sandbox_handle(self._runtime_handle)
            self._surfaces = build_remote_surfaces(self._contract, normalized_handle)
            self._runtime_handle["surface_names"] = tuple(self._surfaces.keys())
            self._runtime_handle = serialize_handle(normalized_handle)
            self._runtime_handle.setdefault("remote_mode", self._contract.mode)
            self._runtime_handle.setdefault("surface_names", tuple(self._surfaces.keys()))
        self._started = True
        return dict(self._runtime_handle)

    def stop(self) -> None:
        self._teardown_sandbox()
        self._started = False
        self._runtime_handle = {}
        self._surfaces = {}

    def create(self) -> dict:
        """Alias for start() for managed lifecycle callers."""

        return self.start()

    def delete(self) -> None:
        """Alias for stop() for managed lifecycle callers."""

        self.stop()

    def exec(
        self,
        command: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        timeout_sec: int = 30,
    ) -> ExecResult:
        sandbox = self._ensure_started()
        return sandbox.exec(_compose_shell_command(command, cwd=cwd, env=env), timeout=timeout_sec)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        self.write_file(remote_path, Path(local_path).read_bytes())

    def download_file(self, remote_path: str, local_path: str) -> None:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(self.read_file(remote_path))

    def read_file(self, remote_path: str) -> bytes:
        sandbox = self._ensure_started()
        return sandbox.read_file(remote_path)

    def write_file(self, remote_path: str, content: bytes) -> None:
        sandbox = self._ensure_started()
        sandbox.write_file(remote_path, content)

    def probe(self, timeout_s: float | None = None) -> bool:
        sandbox = self._ensure_sandbox()
        checker = getattr(sandbox, "is_alive", None)
        if callable(checker):
            return bool(checker(timeout_s=timeout_s))
        return True

    def renew(self, ttl_s: int | None = None) -> None:
        sandbox = self._ensure_sandbox()
        renewer = getattr(sandbox, "renew", None)
        if callable(renewer):
            renewer(ttl_s=ttl_s)

    def attach(self) -> dict:
        return self.start()

    def detach(self) -> None:
        self.stop()

    def runtime_handle(self) -> dict[str, Any]:
        return dict(self._runtime_handle)

    def surfaces(self) -> dict[str, ClientSurface]:
        return dict(self._surfaces)

    @property
    def contract(self) -> Optional[RemoteSandboxContract]:
        return self._contract

    def _ensure_sandbox(self):
        if self._sandbox is not None:
            return self._sandbox
        sandbox_cls = self._sandbox_cls or _load_remote_sandbox_class()
        self._sandbox = sandbox_cls(
            runtime_configs=dict(self._runtime_configs),
            resources=dict(self._resources),
        )
        return self._sandbox

    def _ensure_started(self):
        if not self._started:
            self.start()
        return self._ensure_sandbox()

    def _teardown_sandbox(self) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            return
        teardown = getattr(sandbox, "teardown", None) or getattr(sandbox, "stop", None)
        if callable(teardown):
            teardown()

    def _build_start_config(self) -> dict[str, Any]:
        config = dict(self._config)
        runtime_configs = dict(self._runtime_configs)
        config["runtime_configs"] = runtime_configs
        if self._contract is None:
            return config
        validate_remote_sandbox_contract(self._contract)
        if self._contract.control_endpoint:
            config.setdefault("control_endpoint", self._contract.control_endpoint)
        if self._contract.exec_endpoint:
            config.setdefault("exec_url", self._contract.exec_endpoint)
            config.setdefault("data_endpoint", _derive_base_endpoint(self._contract.exec_endpoint))
        if self._contract.file_endpoint:
            config.setdefault("data_endpoint", self._contract.file_endpoint)
        if self._contract.file_read_url:
            config.setdefault("file_read_url", self._contract.file_read_url)
        if self._contract.file_write_url:
            config.setdefault("file_write_url", self._contract.file_write_url)
        if self._contract.env_endpoint:
            config.setdefault("env_endpoint", self._contract.env_endpoint)
        if self._contract.apis_endpoint:
            config.setdefault("apis_endpoint", self._contract.apis_endpoint)
        if self._contract.mcp_endpoint:
            config.setdefault("mcp_endpoint", self._contract.mcp_endpoint)
        if self._contract.attach_target:
            runtime_configs.setdefault("attach_target", self._contract.attach_target)
            config.setdefault("attach_target", self._contract.attach_target)
        if self._contract.workspace_root:
            runtime_configs.setdefault("workspace_root", self._contract.workspace_root)
            config.setdefault("workspace_root", self._contract.workspace_root)
        if self._contract.provider:
            config.setdefault("provider", self._contract.provider)
        if self._contract.sandbox_id:
            config.setdefault("sandbox_id", self._contract.sandbox_id)
        if self._contract.image:
            config.setdefault("image", self._contract.image)
        runtime_configs.setdefault("remote_mode", self._contract.mode)
        if self._contract.auth:
            if isinstance(self._contract.auth.get("headers"), Mapping):
                runtime_configs["headers"] = dict(self._contract.auth["headers"])
            if self._contract.auth.get("type") is not None:
                runtime_configs.setdefault("auth_type", self._contract.auth["type"])
            if self._contract.auth.get("token") is not None:
                runtime_configs.setdefault("auth_token", self._contract.auth["token"])
            if self._contract.auth.get("api_key_header") is not None:
                runtime_configs.setdefault("api_key_header", self._contract.auth["api_key_header"])
        if self._contract.timeouts:
            if self._contract.timeouts.get("exec_timeout_s") is not None:
                runtime_configs.setdefault("timeout_s", self._contract.timeouts["exec_timeout_s"])
            if self._contract.timeouts.get("file_timeout_s") is not None:
                runtime_configs.setdefault("file_timeout_s", self._contract.timeouts["file_timeout_s"])
            if self._contract.timeouts.get("startup_timeout_s") is not None:
                runtime_configs.setdefault("startup_timeout_s", self._contract.timeouts["startup_timeout_s"])
        if self._contract.retries:
            if self._contract.retries.get("max_retries") is not None:
                runtime_configs.setdefault("max_retries", self._contract.retries["max_retries"])
            if self._contract.retries.get("retry_backoff_factor") is not None:
                runtime_configs.setdefault(
                    "retry_backoff_factor",
                    self._contract.retries["retry_backoff_factor"],
                )
        if self._contract.resources:
            config.setdefault("resources", dict(self._contract.resources))
        if self._contract.params:
            config.setdefault("params", dict(self._contract.params))
        return config


def _load_remote_sandbox_class():
    from gage_eval.sandbox.remote_runtime import RemoteSandbox

    return RemoteSandbox


def _compose_shell_command(
    command: str,
    *,
    cwd: Optional[str],
    env: Optional[Mapping[str, str]],
) -> str:
    parts: list[str] = []
    if env:
        for key, value in env.items():
            parts.append(f"{key}={shlex.quote(str(value))}")
    parts.append(command)
    shell_command = " ".join(parts)
    if cwd:
        shell_command = f"cd {shlex.quote(str(cwd))} && {shell_command}"
    return shell_command


def _derive_base_endpoint(url: str) -> str:
    from urllib.parse import urlsplit, urlunsplit

    split = urlsplit(url)
    path = split.path.rstrip("/")
    if not path:
        return url
    if "/" not in path:
        new_path = ""
    else:
        new_path = path.rsplit("/", 1)[0]
    return urlunsplit((split.scheme, split.netloc, new_path, split.query, split.fragment))
