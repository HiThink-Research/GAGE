"""Docker-backed sandbox runtime implementation (lightweight)."""

from __future__ import annotations

import re
import shlex
import shutil
import socket
import subprocess
import time
import uuid
from http.client import HTTPException
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from gage_eval.sandbox.base import BaseSandbox, ExecResult, SandboxOptionalMixin

_HOST_GATEWAY = "host.docker.internal:host-gateway"
_NETWORK_ALIASES = {"bridge_host", "host_gateway"}
_DEFAULT_STARTUP_TIMEOUT_S = 30
_DEFAULT_STARTUP_INTERVAL_S = 0.2
_DEFAULT_STOP_TIMEOUT_S = 10
_CONTAINER_NAME_SUFFIX_LEN = 8
_CONTAINER_NAME_MAX_LEN = 63
_CONTAINER_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


class DockerSandbox(SandboxOptionalMixin, BaseSandbox):
    """Docker sandbox runtime with optional container lifecycle management."""

    def __init__(
        self,
        runtime_configs: Optional[Dict[str, Any]] = None,
        resources: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._runtime_configs = dict(runtime_configs or {})
        self._resources = dict(resources or {})
        self._config: Dict[str, Any] = {}
        self._running = False
        self._container_id: Optional[str] = None
        self._container_name: Optional[str] = None

    def start(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._config = dict(config or {})
        merged_runtime = dict(self._runtime_configs)
        merged_runtime.update(self._config.get("runtime_configs", {}) or {})
        self._runtime_configs = normalize_runtime_configs(merged_runtime)

        image = self._resolve_image()
        start_container = bool(self._runtime_configs.get("start_container", True))
        start_runner = self._runtime_configs.get("start_runner")
        if callable(start_runner):
            result = start_runner(self._config, self._runtime_configs, dict(self._resources))
            runtime_handle = self._apply_start_result(result)
            self._running = True
            self._wait_for_ready(runtime_handle)
            return runtime_handle

        if not image or not start_container:
            self._running = True
            return self._build_runtime_handle()

        docker_bin = str(self._runtime_configs.get("docker_bin") or "docker")
        _ensure_docker_available(docker_bin)
        self._container_name = self._resolve_container_name()
        command = build_docker_run_command(
            image=image,
            container_name=self._container_name,
            runtime_configs=self._runtime_configs,
            resources=self._resources,
        )
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            error = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"docker_run_failed: {error}")
        self._container_id = completed.stdout.strip() or None
        if not self._container_id:
            raise RuntimeError("docker_run_missing_container_id")
        self._running = True
        runtime_handle = self._build_runtime_handle()
        try:
            self._wait_for_ready(runtime_handle)
        except Exception:
            self.teardown()
            raise
        return runtime_handle

    def exec(self, command: str, timeout: int = 30) -> ExecResult:
        start = time.perf_counter()
        runner = self._runtime_configs.get("command_runner")
        if callable(runner):
            result = runner(command, timeout)
            return _normalize_exec_result(result)
        if not self._container_id:
            raise RuntimeError("docker_container_unavailable")
        docker_bin = str(self._runtime_configs.get("docker_bin") or "docker")
        _ensure_docker_available(docker_bin)
        exec_args = [docker_bin, "exec"]
        exec_user = self._runtime_configs.get("exec_user")
        exec_workdir = self._runtime_configs.get("exec_workdir")
        if exec_user:
            exec_args.extend(["-u", str(exec_user)])
        if exec_workdir:
            exec_args.extend(["-w", str(exec_workdir)])
        exec_args.extend([self._container_id, "/bin/sh", "-lc", command])
        completed = subprocess.run(
            exec_args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return ExecResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=elapsed_ms,
        )

    def teardown(self) -> None:
        docker_bin = str(self._runtime_configs.get("docker_bin") or "docker")
        if (self._container_id or self._container_name) and _docker_available(docker_bin):
            stop_timeout = int(self._runtime_configs.get("stop_timeout_s", _DEFAULT_STOP_TIMEOUT_S))
            target = self._container_id or self._container_name
            if target:
                subprocess.run(
                    [docker_bin, "stop", "-t", str(stop_timeout), target],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                auto_remove = bool(self._runtime_configs.get("auto_remove", True))
                remove_container = bool(self._runtime_configs.get("remove_container", True))
                lifecycle = str(self._config.get("lifecycle", "per_sample"))
                if lifecycle == "per_sample":
                    remove_container = True
                if remove_container and (lifecycle == "per_sample" or not auto_remove):
                    subprocess.run(
                        [docker_bin, "rm", "-f", target],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
        self._running = False
        self._container_id = None
        self._container_name = None

    def is_alive(self, timeout_s: float | None = None) -> bool:
        if not self._running:
            return False
        if not self._container_id:
            return True
        docker_bin = str(self._runtime_configs.get("docker_bin") or "docker")
        if not _docker_available(docker_bin):
            return self._running
        timeout = 2 if timeout_s is None else max(0.1, float(timeout_s))
        completed = subprocess.run(
            [docker_bin, "inspect", "-f", "{{.State.Running}}", self._container_id],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            return False
        return completed.stdout.strip().lower() == "true"

    def _resolve_container_name(self) -> str:
        explicit = (
            self._runtime_configs.get("container_name")
            or self._runtime_configs.get("name")
            or self._config.get("container_name")
            or self._config.get("name")
        )
        if explicit:
            return str(explicit)
        prefix = str(
            self._runtime_configs.get("container_name_prefix")
            or self._config.get("container_name_prefix")
            or "gage-sandbox"
        )
        suffix = (
            self._runtime_configs.get("container_name_suffix")
            or self._config.get("container_name_suffix")
        )
        if suffix:
            prefix = f"{prefix}-{suffix}"
        prefix = _sanitize_container_name(prefix)
        max_prefix_len = _CONTAINER_NAME_MAX_LEN - _CONTAINER_NAME_SUFFIX_LEN - 1
        if max_prefix_len > 0 and len(prefix) > max_prefix_len:
            prefix = prefix[:max_prefix_len].rstrip("-.") or "sandbox"
        return f"{prefix}-{uuid.uuid4().hex[:_CONTAINER_NAME_SUFFIX_LEN]}"

    def _resolve_image(self) -> Optional[str]:
        return self._config.get("image") or self._runtime_configs.get("image")

    def _build_runtime_handle(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        handle: Dict[str, Any] = {}
        if self._container_id:
            handle["container_id"] = self._container_id
        if self._container_name:
            handle["container_name"] = self._container_name
        for key in (
            "env_endpoint",
            "environment_endpoint",
            "env_url",
            "environment_url",
            "apis_endpoint",
            "apis_url",
            "mcp_endpoint",
            "mcp_url",
        ):
            value = self._runtime_configs.get(key) or self._config.get(key)
            if value:
                handle[key] = value
        if extra:
            handle.update(extra)
        return handle

    def _apply_start_result(self, result: Any) -> Dict[str, Any]:
        runtime_handle: Dict[str, Any] = {}
        container_id, container_name = _extract_container_identity(result)
        if container_id:
            self._container_id = container_id
        if container_name:
            self._container_name = container_name
        if isinstance(result, dict):
            runtime_handle = dict(result.get("runtime_handle") or {})
        return self._build_runtime_handle(extra=runtime_handle)

    def _wait_for_ready(self, runtime_handle: Dict[str, Any]) -> None:
        if not bool(self._runtime_configs.get("wait_for_ready", True)):
            return
        timeout_s = float(self._runtime_configs.get("startup_timeout_s", _DEFAULT_STARTUP_TIMEOUT_S))
        interval_s = float(self._runtime_configs.get("startup_interval_s", _DEFAULT_STARTUP_INTERVAL_S))
        endpoints = _resolve_wait_endpoints(self._runtime_configs, runtime_handle)
        ports = _resolve_wait_ports(self._runtime_configs)
        http_endpoints = _normalize_list(self._runtime_configs.get("wait_for_http_endpoints"))
        if not endpoints and not ports and not http_endpoints:
            return
        if endpoints:
            for endpoint in endpoints:
                host_port = _parse_endpoint(endpoint)
                if host_port:
                    _wait_for_tcp(host_port[0], host_port[1], timeout_s, interval_s)
        if ports:
            for port in ports:
                _wait_for_tcp("127.0.0.1", port, timeout_s, interval_s)
        if http_endpoints:
            for endpoint in http_endpoints:
                _wait_for_http(str(endpoint), timeout_s, interval_s)


def normalize_runtime_configs(runtime_configs: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Docker runtime configs, including network mappings."""

    normalized = dict(runtime_configs or {})
    network_mode = normalized.get("network_mode")
    if network_mode in _NETWORK_ALIASES:
        normalized["network_mode"] = "bridge"
        normalized["extra_hosts"] = _merge_extra_hosts(normalized.get("extra_hosts"), _HOST_GATEWAY)
    elif network_mode == "host":
        normalized["network_mode"] = "host"
    elif network_mode == "none":
        normalized["network_mode"] = "none"
    return normalized


def _merge_extra_hosts(raw: Any, entry: str) -> List[str]:
    extra_hosts: List[str] = []
    if isinstance(raw, dict):
        for host, target in raw.items():
            extra_hosts.append(f"{host}:{target}")
    elif isinstance(raw, list):
        extra_hosts = [str(item) for item in raw]
    if entry not in extra_hosts:
        extra_hosts.append(entry)
    return extra_hosts


def _sanitize_container_name(value: str) -> str:
    cleaned = _CONTAINER_NAME_PATTERN.sub("-", value.strip())
    cleaned = cleaned.strip("-.")
    return cleaned or "sandbox"


def build_docker_run_command(
    *,
    image: str,
    container_name: Optional[str],
    runtime_configs: Dict[str, Any],
    resources: Dict[str, Any],
) -> List[str]:
    runtime_configs = dict(runtime_configs or {})
    docker_bin = str(runtime_configs.get("docker_bin") or "docker")
    args: List[str] = [docker_bin, "run"]
    if bool(runtime_configs.get("detach", True)):
        args.append("-d")
    if bool(runtime_configs.get("auto_remove", True)):
        args.append("--rm")
    if container_name:
        args.extend(["--name", container_name])
    platform = runtime_configs.get("platform")
    if platform:
        args.extend(["--platform", str(platform)])
    network_mode = runtime_configs.get("network_mode")
    if network_mode:
        args.extend(["--network", str(network_mode)])
    _apply_resource_limits(args, runtime_configs, resources)
    if network_mode != "host":
        for port in _normalize_ports(runtime_configs.get("ports")):
            args.extend(["-p", port])
    for host in _normalize_extra_hosts(runtime_configs.get("extra_hosts")):
        args.extend(["--add-host", host])
    for env in _normalize_env(runtime_configs.get("env")):
        args.extend(["-e", env])
    for env_file in _normalize_list(runtime_configs.get("env_file")):
        args.extend(["--env-file", env_file])
    for volume in _normalize_volumes(runtime_configs.get("volumes")):
        args.extend(["-v", volume])
    workdir = runtime_configs.get("workdir") or runtime_configs.get("working_dir")
    if workdir:
        args.extend(["-w", str(workdir)])
    user = runtime_configs.get("user")
    if user:
        args.extend(["-u", str(user)])
    entrypoint = runtime_configs.get("entrypoint")
    if entrypoint:
        entrypoint_value = _normalize_command(entrypoint)
        if entrypoint_value:
            args.extend(["--entrypoint", entrypoint_value[0]])
    for extra in _normalize_command(runtime_configs.get("extra_args")):
        args.append(extra)
    args.append(image)
    for command in _normalize_command(runtime_configs.get("command") or runtime_configs.get("cmd")):
        args.append(command)
    return args


def _apply_resource_limits(
    args: List[str],
    runtime_configs: Dict[str, Any],
    resources: Dict[str, Any],
) -> None:
    cpu = runtime_configs.get("cpus") or resources.get("cpu") or resources.get("cpus")
    if cpu is not None:
        args.extend(["--cpus", str(cpu)])
    memory = (
        runtime_configs.get("memory")
        or runtime_configs.get("mem_limit")
        or resources.get("memory")
        or resources.get("mem_limit")
    )
    if memory is not None:
        args.extend(["--memory", str(memory)])
    cpu_quota = runtime_configs.get("cpu_quota")
    if cpu_quota is not None:
        args.extend(["--cpu-quota", str(cpu_quota)])
    cpuset_cpus = runtime_configs.get("cpuset_cpus")
    if cpuset_cpus:
        args.extend(["--cpuset-cpus", str(cpuset_cpus)])
    shm_size = runtime_configs.get("shm_size")
    if shm_size:
        args.extend(["--shm-size", str(shm_size)])


def _normalize_ports(raw: Any) -> List[str]:
    if raw is None:
        return []
    ports: List[str] = []
    if isinstance(raw, dict):
        for host, container in raw.items():
            if container is None:
                ports.append(str(host))
            else:
                ports.append(f"{host}:{container}")
        return ports
    if isinstance(raw, (list, tuple)):
        for entry in raw:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                ports.append(f"{entry[0]}:{entry[1]}")
            else:
                ports.append(str(entry))
        return ports
    return [str(raw)]


def _normalize_env(raw: Any) -> List[str]:
    if raw is None:
        return []
    envs: List[str] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            if value is None:
                envs.append(str(key))
            else:
                envs.append(f"{key}={value}")
        return envs
    if isinstance(raw, (list, tuple)):
        for entry in raw:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                envs.append(f"{entry[0]}={entry[1]}")
            else:
                envs.append(str(entry))
        return envs
    return [str(raw)]


def _normalize_volumes(raw: Any) -> List[str]:
    if raw is None:
        return []
    volumes: List[str] = []
    if isinstance(raw, dict):
        for host, container in raw.items():
            if container is None:
                volumes.append(str(host))
            else:
                volumes.append(f"{host}:{container}")
        return volumes
    if isinstance(raw, (list, tuple)):
        for entry in raw:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                volumes.append(f"{entry[0]}:{entry[1]}")
            else:
                volumes.append(str(entry))
        return volumes
    return [str(raw)]


def _normalize_extra_hosts(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [f"{host}:{target}" for host, target in raw.items()]
    if isinstance(raw, (list, tuple)):
        return [str(entry) for entry in raw]
    return [str(raw)]


def _normalize_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    return [str(raw)]


def _normalize_command(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return shlex.split(raw)
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if item is not None]
    return [str(raw)]


def _resolve_wait_endpoints(
    runtime_configs: Dict[str, Any],
    runtime_handle: Dict[str, Any],
) -> List[str]:
    override = runtime_configs.get("wait_for_endpoints")
    if override is not None:
        return _normalize_list(override)
    endpoints: List[str] = []
    for key in (
        "env_endpoint",
        "environment_endpoint",
        "env_url",
        "environment_url",
        "apis_endpoint",
        "apis_url",
        "mcp_endpoint",
        "mcp_url",
    ):
        value = runtime_handle.get(key) or runtime_configs.get(key)
        if value:
            endpoints.append(str(value))
    return endpoints


def _resolve_wait_ports(runtime_configs: Dict[str, Any]) -> List[int]:
    override = runtime_configs.get("wait_for_ports")
    if override is not None:
        return _normalize_ports_to_ints(override)
    port_mappings = _normalize_ports(runtime_configs.get("ports"))
    return _normalize_ports_to_ints(port_mappings)


def _normalize_ports_to_ints(raw: Any) -> List[int]:
    ports: List[int] = []
    for entry in _normalize_ports(raw):
        port = _extract_host_port(entry)
        if port is not None:
            ports.append(port)
    return ports


def _extract_host_port(mapping: str) -> Optional[int]:
    if not mapping:
        return None
    raw = str(mapping)
    if "/" in raw:
        raw = raw.split("/", 1)[0]
    parts = raw.split(":")
    if len(parts) == 1:
        candidate = parts[0]
    elif len(parts) >= 3:
        candidate = parts[-2]
    else:
        candidate = parts[0]
    try:
        return int(candidate)
    except ValueError:
        return None


def _parse_endpoint(endpoint: str) -> Optional[Tuple[str, int]]:
    raw = endpoint.strip()
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    host = parsed.hostname
    if not host:
        return None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def _wait_for_tcp(host: str, port: int, timeout_s: float, interval_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=interval_s):
                return
        except OSError:
            time.sleep(interval_s)
    raise RuntimeError(f"docker_wait_timeout:{host}:{port}")


def _wait_for_http(endpoint: str, timeout_s: float, interval_s: float) -> None:
    url = endpoint.strip()
    if not url:
        return
    if "://" not in url:
        url = f"http://{url}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            request = Request(url, method="GET")
            with urlopen(request, timeout=interval_s) as response:
                if response.status < 500:
                    return
        except HTTPError as exc:
            if exc.code < 500:
                return
            time.sleep(interval_s)
            continue
        except (ConnectionResetError, HTTPException, OSError, TimeoutError, URLError):
            time.sleep(interval_s)
            continue
        time.sleep(interval_s)
    raise RuntimeError(f"docker_wait_http_timeout:{url}")


def _ensure_docker_available(docker_bin: str) -> None:
    if not _docker_available(docker_bin):
        raise RuntimeError(f"docker_binary_not_found:{docker_bin}")


def _docker_available(docker_bin: str) -> bool:
    return shutil.which(docker_bin) is not None


def _normalize_exec_result(raw: Any) -> ExecResult:
    if isinstance(raw, ExecResult):
        return raw
    if isinstance(raw, dict):
        return ExecResult(
            exit_code=int(raw.get("exit_code", 0)),
            stdout=str(raw.get("stdout", "")),
            stderr=str(raw.get("stderr", "")),
            duration_ms=float(raw.get("duration_ms", 0.0)),
        )
    return ExecResult(exit_code=0, stdout=str(raw), stderr="", duration_ms=0.0)


def _extract_container_identity(result: Any) -> Tuple[Optional[str], Optional[str]]:
    if isinstance(result, dict):
        container_id = result.get("container_id") or result.get("id")
        container_name = result.get("container_name") or result.get("name")
        return (
            str(container_id) if container_id else None,
            str(container_name) if container_name else None,
        )
    if isinstance(result, str):
        cleaned = result.strip()
        return (cleaned or None), None
    return None, None
