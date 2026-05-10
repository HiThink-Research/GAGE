"""Docker-backed AgentKit v2 environment provider."""

from __future__ import annotations

import asyncio
import io
import posixpath
import shlex
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from gage_eval.environment.contracts import (
    DEFAULT_READ_FILE_LIMIT_BYTES,
    BaseEnvironment,
    ExecResult,
    FileInfo,
    validate_read_size,
)
from gage_eval.environment.errors import (
    EnvironmentAttachError,
    EnvironmentCreateError,
    EnvironmentExecError,
    EnvironmentFileNotFoundError,
    EnvironmentPreflightError,
    EnvironmentTimeoutError,
    EnvironmentTransferError,
)
from gage_eval.environment.profiles import EnvironmentProfile
from gage_eval.environment.resources import EnvironmentCapabilities, EnvironmentResources

from .config import DockerEnvironmentConfig, DockerMount


class DockerEnvironmentProvider:
    """Registry-facing provider with an injectable Docker client adapter."""

    retry_budget_by_failure = {EnvironmentCreateError: 2}

    def __init__(self, *, client: Any | None = None) -> None:
        self._client = client

    async def preflight(
        self,
        *,
        kit_id: str,
        provider: str,
        profile_id: str,
        profile: EnvironmentProfile,
        provider_config: Any,
        resources: EnvironmentResources,
        startup_env: dict[str, str],
        lifecycle: str,
        metadata: dict[str, Any],
    ) -> None:
        del kit_id, provider, profile_id, resources, startup_env, lifecycle, metadata
        _coerce_config(provider_config, profile=profile)
        client = self._client
        if client is not None and callable(getattr(client, "ping", None)):
            try:
                await _maybe_to_thread(client.ping)
            except Exception as exc:
                raise EnvironmentPreflightError(f"docker.preflight ping failed: {exc}") from exc

    async def create(
        self,
        *,
        kit_id: str,
        provider: str,
        profile_id: str,
        profile: EnvironmentProfile,
        provider_config: Any,
        resources: EnvironmentResources,
        startup_env: dict[str, str],
        lifecycle: str,
        metadata: dict[str, Any],
    ) -> BaseEnvironment:
        del kit_id, provider, lifecycle
        config = _coerce_config(provider_config, profile=profile)
        client = self._client or _default_docker_client()
        env_id = f"docker-{uuid4().hex}"
        container_name = _container_name(config, env_id)
        environment = {**config.environment, **startup_env}
        network_policy = config.network_policy if config.network_policy is not None else resources.network_policy
        mounts = _docker_mounts(config)
        try:
            container = await _maybe_to_thread(
                client.create_container,
                image=config.image,
                name=container_name,
                entrypoint=config.entrypoint,
                command=config.keepalive_command,
                detach=True,
                tty=True,
                working_dir=config.workdir,
                user=config.user,
                environment=environment,
                privileged=config.privileged,
                network_policy=network_policy,
                network_mode=_docker_network_mode(config, network_policy),
                ports=_docker_ports(config),
                extra_hosts=list(config.extra_hosts),
                mounts=mounts,
                platform=config.docker_platform,
            )
        except Exception as exc:
            raise EnvironmentCreateError(f"docker.create image={config.image!r}: {exc}") from exc

        return DockerEnvironment(
            env_id=env_id,
            name=container_name,
            client=client,
            container=container,
            config=config,
            network_policy=network_policy,
            metadata={
                "profile_id": profile_id,
                **_runtime_handle_metadata(config),
                **{key: str(value) for key, value in metadata.items() if isinstance(value, str)},
            },
        )

    async def health_check(self, environment: BaseEnvironment) -> bool:
        config = getattr(environment, "_config", None)
        if not isinstance(config, DockerEnvironmentConfig) or not config.wait_for_http_endpoints:
            return True
        deadline = time.monotonic() + float(config.startup_timeout_s or 0)
        while True:
            if await _http_endpoints_available(config.wait_for_http_endpoints, timeout_s=config.startup_interval_s):
                return True
            if config.startup_timeout_s <= 0 or time.monotonic() >= deadline:
                return False
            await asyncio.sleep(float(config.startup_interval_s))


class DockerEnvironment:
    env_id: str
    name: str
    provider = "docker"

    def __init__(
        self,
        *,
        env_id: str,
        name: str,
        client: Any,
        container: Any,
        config: DockerEnvironmentConfig,
        network_policy: str,
        metadata: dict[str, str],
    ) -> None:
        self.env_id = env_id
        self.name = name
        self.metadata = metadata
        self._client = client
        self._container = container
        self._config = config
        self._network_policy = network_policy
        self.capabilities = EnvironmentCapabilities(
            supports_mounts=config.use_host_workdir_mount,
            supports_upload_download=True,
            supports_internet_control=network_policy != "block",
            supports_privileged_dind=config.privileged,
            default_user=config.user,
        )

    async def start(self, *, force_build: bool = False) -> None:
        if not force_build:
            return
        build_image = getattr(self._client, "build_image", None)
        if callable(build_image):
            try:
                await _maybe_to_thread(build_image, image=self._config.image, platform=self._config.docker_platform)
            except Exception as exc:
                raise EnvironmentCreateError(f"docker.build image={self._config.image!r}: {exc}") from exc

    async def attach(self) -> None:
        if self._container is None:
            raise EnvironmentAttachError(f"docker.attach env_id={self.env_id}")

    async def stop(self, *, delete: bool = True) -> None:
        try:
            await _maybe_to_thread(self._container.stop, delete=delete)
        except AttributeError:
            return
        except Exception as exc:
            raise EnvironmentCreateError(f"docker.stop env_id={self.env_id}: {exc}") from exc

    async def exec(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_s: int | None = None,
        user: str | None = None,
        shell: Literal["sh", "login", "none"] = "sh",
    ) -> ExecResult:
        docker_command = _docker_exec_command(command, shell=shell)
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                _maybe_to_thread(
                    self._container.exec,
                    command=docker_command,
                    env=env,
                    cwd=cwd or self._config.exec_workdir,
                    timeout_s=timeout_s,
                    user=user,
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise EnvironmentTimeoutError(f"docker.exec timeout_s={timeout_s} command={command!r}") from exc
        except Exception as exc:
            raise EnvironmentExecError(f"docker.exec command={command!r}: {exc}") from exc
        return _exec_result(result, command=command, duration_s=time.monotonic() - start)

    async def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        source = Path(local_path)
        if not source.is_file():
            raise EnvironmentFileNotFoundError(f"local file not found: {source}")
        host_path = self._mounted_host_path(remote_path, for_write=True)
        if host_path is not None:
            try:
                host_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, host_path)
                return
            except Exception as exc:
                raise _mounted_transfer_error("upload_file", remote_path, exc) from exc
        await self._put_tar_file(source.read_bytes(), remote_path)

    async def upload_dir(self, local_path: str | Path, remote_path: str) -> None:
        source = Path(local_path)
        if not source.is_dir():
            raise EnvironmentFileNotFoundError(f"local dir not found: {source}")
        host_path = self._mounted_host_path(remote_path, for_write=True)
        if host_path is not None:
            try:
                if host_path.exists():
                    shutil.rmtree(host_path)
                shutil.copytree(source, host_path)
                return
            except Exception as exc:
                raise _mounted_transfer_error("upload_dir", remote_path, exc) from exc
        await self._put_tar_dir(source, remote_path)

    async def download_file(self, remote_path: str, local_path: str | Path) -> None:
        target = Path(local_path)
        host_path = self._mounted_host_path(remote_path)
        if host_path is not None:
            if not host_path.is_file():
                raise EnvironmentFileNotFoundError(f"remote file not found: {remote_path}")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(host_path, target)
                return
            except Exception as exc:
                raise _mounted_transfer_error("download_file", remote_path, exc) from exc
        data = await self._get_tar(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _extract_single_file(data, target, remote_path=remote_path)

    async def download_dir(self, remote_path: str, local_path: str | Path) -> None:
        target = Path(local_path)
        host_path = self._mounted_host_path(remote_path)
        if host_path is not None:
            if not host_path.is_dir():
                raise EnvironmentFileNotFoundError(f"remote dir not found: {remote_path}")
            try:
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(host_path, target)
                return
            except Exception as exc:
                raise _mounted_transfer_error("download_dir", remote_path, exc) from exc
        data = await self._get_tar(remote_path)
        _extract_tar_tree(data, target)

    async def write_file(self, path: str, content: bytes | str) -> None:
        payload = content.encode("utf-8") if isinstance(content, str) else content
        host_path = self._mounted_host_path(path, for_write=True)
        if host_path is not None:
            try:
                host_path.parent.mkdir(parents=True, exist_ok=True)
                host_path.write_bytes(payload)
                return
            except Exception as exc:
                raise _mounted_transfer_error("write_file", path, exc) from exc
        await self._put_tar_file(payload, path)

    async def read_file(self, path: str, *, max_bytes: int = DEFAULT_READ_FILE_LIMIT_BYTES) -> bytes:
        host_path = self._mounted_host_path(path)
        if host_path is not None:
            if not host_path.is_file():
                raise EnvironmentFileNotFoundError(f"remote file not found: {path}")
            validate_read_size(path, host_path.stat().st_size, max_bytes=max_bytes)
            try:
                return host_path.read_bytes()
            except Exception as exc:
                raise _mounted_transfer_error("read_file", path, exc) from exc
        data = await self._get_tar(path)
        content = _read_single_file_from_tar(data, remote_path=path)
        validate_read_size(path, len(content), max_bytes=max_bytes)
        return content

    async def list_files(self, path: str) -> list[FileInfo]:
        host_path = self._mounted_host_path(path)
        if host_path is None:
            result = await self.exec(
                f"find {_quote_remote_path(path)} -maxdepth 1 -mindepth 1 -printf '%y %s %p\\n'",
                shell="sh",
            )
            return [_file_info_from_find_line(line) for line in result.stdout.splitlines() if line.strip()]
        if not host_path.is_dir():
            raise EnvironmentFileNotFoundError(f"remote dir not found: {path}")
        return [_file_info_from_path(entry, _host_to_remote(entry, host_path, path)) for entry in host_path.iterdir()]

    async def is_file(self, path: str) -> bool:
        host_path = self._mounted_host_path(path)
        if host_path is not None:
            return host_path.is_file()
        result = await self.exec(f"test -f {_quote_remote_path(path)}", shell="sh")
        return result.exit_code == 0

    async def is_dir(self, path: str) -> bool:
        host_path = self._mounted_host_path(path)
        if host_path is not None:
            return host_path.is_dir()
        result = await self.exec(f"test -d {_quote_remote_path(path)}", shell="sh")
        return result.exit_code == 0

    async def get_logs(self, *, stream: Literal["stdout", "stderr"] | None = None) -> str:
        try:
            return str(await _maybe_to_thread(self._container.logs, stream=stream))
        except Exception as exc:
            raise EnvironmentExecError(f"docker.logs env_id={self.env_id}: {exc}") from exc

    async def describe(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "name": self.name,
            "provider": self.provider,
            "capabilities": self.capabilities.model_dump(mode="python"),
            "metadata": dict(self.metadata),
            "diagnostics": {
                "container_id": getattr(self._container, "id", None),
                "container_name": getattr(self._container, "name", None),
                "image": self._config.image,
                "docker_platform": self._config.docker_platform,
                "network_policy": self._network_policy,
                "workdir": self._config.workdir,
                "exec_workdir": self._config.exec_workdir,
                "entrypoint": self._config.entrypoint,
                "keepalive_command": self._config.keepalive_command,
                "mounted_workdir": self._config.use_host_workdir_mount,
                "mount_targets": [mount.target for mount in self._config.mounts],
            },
        }

    async def _put_tar_file(self, content: bytes, remote_path: str) -> None:
        parent = posixpath.dirname(_normalize_remote_path(remote_path)) or "/"
        name = posixpath.basename(remote_path)
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w") as archive:
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        try:
            await self._ensure_remote_dir(parent)
            await _maybe_to_thread(self._container.put_archive, parent, payload.getvalue())
        except Exception as exc:
            raise EnvironmentTransferError(f"docker.put_archive path={remote_path!r}: {exc}") from exc

    async def _put_tar_dir(self, source: Path, remote_path: str) -> None:
        parent = posixpath.dirname(_normalize_remote_path(remote_path)) or "/"
        root_name = posixpath.basename(remote_path.rstrip("/"))
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w") as archive:
            for entry in source.rglob("*"):
                archive.add(entry, arcname=str(Path(root_name) / entry.relative_to(source)))
        try:
            await self._ensure_remote_dir(parent)
            await _maybe_to_thread(self._container.put_archive, parent, payload.getvalue())
        except Exception as exc:
            raise EnvironmentTransferError(f"docker.put_archive path={remote_path!r}: {exc}") from exc

    async def _get_tar(self, remote_path: str) -> bytes:
        try:
            return await _maybe_to_thread(self._container.get_archive, remote_path)
        except FileNotFoundError as exc:
            raise EnvironmentFileNotFoundError(f"remote path not found: {remote_path}") from exc
        except Exception as exc:
            if _is_not_found_exception(exc):
                raise EnvironmentFileNotFoundError(f"remote path not found: {remote_path}") from exc
            raise EnvironmentTransferError(f"docker.get_archive path={remote_path!r}: {exc}") from exc

    async def _ensure_remote_dir(self, remote_dir: str) -> None:
        result = await self.exec(f"mkdir -p {_quote_remote_path(remote_dir)}", shell="sh")
        if result.exit_code != 0:
            raise EnvironmentTransferError(f"docker.mkdir path={remote_dir!r} exit_code={result.exit_code}")

    def _mounted_host_path(self, remote_path: str, *, for_write: bool = False) -> Path | None:
        if not self._config.use_host_workdir_mount or not self._config.host_workdir:
            return None
        normalized = _normalize_remote_path(remote_path)
        workdir = _normalize_remote_path(self._config.workdir)
        if normalized == workdir:
            relative = "."
        elif normalized.startswith(f"{workdir}/"):
            relative = normalized[len(workdir) + 1 :]
        else:
            return None
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise EnvironmentTransferError(f"unsafe mounted path: {remote_path!r}")
        host_root = Path(self._config.host_workdir).resolve(strict=False)
        candidate = Path(self._config.host_workdir) / relative_path
        for existing in _existing_path_chain(Path(self._config.host_workdir), relative_path):
            if existing.is_symlink():
                raise EnvironmentTransferError(f"mounted path contains symlink: {remote_path!r}")
        candidate_resolved = candidate.resolve(strict=False)
        parent_resolved = candidate.parent.resolve(strict=False)
        if not _path_is_relative_to(candidate_resolved, host_root) or not _path_is_relative_to(
            parent_resolved,
            host_root,
        ):
            raise EnvironmentTransferError(f"mounted path escapes host_workdir: {remote_path!r}")
        return candidate_resolved


def _mounted_transfer_error(operation: str, remote_path: str, exc: BaseException) -> EnvironmentTransferError:
    detail = exc.__class__.__name__
    errno = getattr(exc, "errno", None)
    if errno is not None:
        detail = f"{detail} errno={errno}"
    return EnvironmentTransferError(f"docker.{operation} mounted path={remote_path!r} failed: {detail}")


def _existing_path_chain(host_root: Path, relative_path: Path) -> list[Path]:
    current = host_root
    existing = [current]
    for part in relative_path.parts:
        if part == ".":
            continue
        current = current / part
        if current.exists() or current.is_symlink():
            existing.append(current)
    return existing


def _quote_remote_path(path: str) -> str:
    return shlex.quote(_normalize_remote_path(path))


def _is_not_found_exception(exc: BaseException) -> bool:
    return any(cls.__name__ == "NotFound" for cls in type(exc).mro())


def _coerce_config(provider_config: Any, *, profile: EnvironmentProfile) -> DockerEnvironmentConfig:
    if isinstance(provider_config, DockerEnvironmentConfig):
        return provider_config
    if isinstance(provider_config, BaseModel):
        raw = provider_config.model_dump(mode="python", exclude_none=True)
    elif isinstance(provider_config, dict):
        raw = dict(provider_config)
    else:
        raw = {}
    profile_defaults = {key: value for key, value in profile.config.items() if key in DockerEnvironmentConfig.model_fields}
    profile_defaults.update(raw)
    try:
        return DockerEnvironmentConfig.model_validate(profile_defaults)
    except ValidationError as exc:
        raise EnvironmentPreflightError(f"docker.config validation failed: {exc}") from exc


def _container_name(config: DockerEnvironmentConfig, env_id: str) -> str:
    prefix = config.container_name_prefix or "gage-env"
    return f"{prefix}-{env_id[-12:]}"


def _docker_mounts(config: DockerEnvironmentConfig) -> list[dict[str, Any]]:
    mounts = [mount.model_dump(mode="python") for mount in config.mounts]
    if config.use_host_workdir_mount and config.host_workdir:
        mounts.append(
            {
                "type": "bind",
                "source": config.host_workdir,
                "target": config.workdir,
                "read_only": False,
            }
        )
    return mounts


def _network_mode(network_policy: str) -> str | None:
    if network_policy == "block":
        return "none"
    if network_policy in {"allow", "egress_only"}:
        return "bridge"
    return None


def _docker_network_mode(config: DockerEnvironmentConfig, network_policy: str) -> str | None:
    if config.network_mode:
        return config.network_mode
    return _network_mode(network_policy)


def _docker_ports(config: DockerEnvironmentConfig) -> dict[str, int | None]:
    ports: dict[str, int | None] = {}
    for mapping in config.ports:
        host, container = _parse_port_mapping(mapping)
        if not container:
            continue
        ports[f"{container}/tcp"] = host
    return ports


def _parse_port_mapping(mapping: str) -> tuple[int | None, int | None]:
    parts = str(mapping).split(":")
    if len(parts) == 1:
        return None, _coerce_port(parts[0])
    return _coerce_port(parts[-2]), _coerce_port(parts[-1])


def _coerce_port(value: Any) -> int | None:
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return port if 0 < port <= 65535 else None


def _runtime_handle_metadata(config: DockerEnvironmentConfig) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key in (
        "env_endpoint",
        "environment_endpoint",
        "apis_endpoint",
        "mcp_endpoint",
        "env_url",
        "apis_url",
        "mcp_url",
    ):
        value = config.environment.get(key)
        if isinstance(value, str) and value:
            metadata[key] = value
    return metadata


async def _http_endpoints_available(endpoints: list[str], *, timeout_s: float) -> bool:
    for endpoint in endpoints:
        if not await _http_endpoint_available(endpoint, timeout_s=timeout_s):
            return False
    return True


async def _http_endpoint_available(endpoint: str, *, timeout_s: float) -> bool:
    def _probe() -> bool:
        try:
            request = urllib.request.Request(
                endpoint,
                headers={"Accept": "application/json, text/event-stream, */*"},
            )
            with urllib.request.urlopen(request, timeout=max(0.1, float(timeout_s))) as response:
                return int(getattr(response, "status", 200)) < 500
        except Exception:
            return False

    return await asyncio.to_thread(_probe)


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _docker_exec_command(command: str, *, shell: Literal["sh", "login", "none"]) -> str | list[str]:
    if shell == "none":
        return command
    if shell == "login":
        return ["/bin/sh", "-lc", f". /etc/profile >/dev/null 2>&1 || true; {command}"]
    return ["/bin/sh", "-lc", command]


def _exec_result(result: Any, *, command: str, duration_s: float) -> ExecResult:
    if isinstance(result, ExecResult):
        return result.model_copy(update={"duration_s": result.duration_s if result.duration_s is not None else duration_s})
    if isinstance(result, dict):
        return ExecResult(
            command=command,
            exit_code=result.get("exit_code"),
            stdout=_decode_output(result.get("stdout", "")),
            stderr=_decode_output(result.get("stderr", "")),
            duration_s=result.get("duration_s", duration_s),
            timed_out=bool(result.get("timed_out", False)),
            truncated=bool(result.get("truncated", False)),
            metadata=dict(result.get("metadata") or {}),
        )
    if isinstance(result, tuple) and len(result) == 2:
        exit_code, output = result
        return ExecResult(command=command, exit_code=exit_code, stdout=_decode_output(output), duration_s=duration_s)
    raise EnvironmentExecError(f"docker.exec unexpected result type={type(result).__name__}")


def _decode_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, tuple):
        stdout, stderr = value
        return _decode_output(stdout) + _decode_output(stderr)
    return str(value)


async def _maybe_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)


def _normalize_remote_path(path: str) -> str:
    if not path.startswith("/"):
        raise EnvironmentTransferError(f"remote path must be absolute: {path!r}")
    normalized = posixpath.normpath(path)
    if normalized == ".":
        return "/"
    return normalized


def _read_single_file_from_tar(data: bytes, *, remote_path: str) -> bytes:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        for member in archive.getmembers():
            if member.isfile():
                extracted = archive.extractfile(member)
                if extracted is None:
                    break
                return extracted.read()
    raise EnvironmentFileNotFoundError(f"remote file not found: {remote_path}")


def _extract_single_file(data: bytes, target: Path, *, remote_path: str) -> None:
    content = _read_single_file_from_tar(data, remote_path=remote_path)
    target.write_bytes(content)


def _extract_tar_tree(data: bytes, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        for member in archive.getmembers():
            destination = target / member.name
            destination.resolve().relative_to(target.resolve())
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(extracted.read())


def _file_info_from_path(path: Path, remote_path: str) -> FileInfo:
    if path.is_file():
        kind = "file"
    elif path.is_dir():
        kind = "dir"
    elif path.is_symlink():
        kind = "symlink"
    else:
        kind = "other"
    return FileInfo(path=remote_path, kind=kind, size_bytes=path.stat().st_size if path.is_file() else None)


def _host_to_remote(entry: Path, host_root: Path, remote_root: str) -> str:
    return posixpath.join(_normalize_remote_path(remote_root), entry.relative_to(host_root).as_posix())


def _file_info_from_find_line(line: str) -> FileInfo:
    kind_code, size, path = line.split(" ", 2)
    kind = {"f": "file", "d": "dir", "l": "symlink"}.get(kind_code, "other")
    return FileInfo(path=path, kind=kind, size_bytes=int(size) if kind == "file" else None)


def _default_docker_client() -> Any:
    try:
        import docker
    except ImportError as exc:
        raise EnvironmentCreateError("docker.sdk missing; install the docker Python package") from exc
    return _DockerSDKClientAdapter(docker.from_env())


class _DockerSDKClientAdapter:
    def __init__(self, client: Any) -> None:
        self._client = client

    def ping(self) -> None:
        self._client.ping()

    def create_container(self, **kwargs: Any) -> Any:
        volumes = _sdk_volumes(kwargs.pop("mounts"))
        network_mode = kwargs.pop("network_mode")
        kwargs.pop("network_policy", None)
        if network_mode is not None:
            kwargs["network_mode"] = network_mode
        return _DockerSDKContainerAdapter(self._client.containers.run(volumes=volumes, **kwargs))


class _DockerSDKContainerAdapter:
    def __init__(self, container: Any) -> None:
        self._container = container

    @property
    def id(self) -> str:
        return str(self._container.id)

    @property
    def name(self) -> str:
        return str(self._container.name)

    def exec(self, *, command: str | list[str], env: dict[str, str] | None, cwd: str | None, timeout_s: int | None, user: str | None) -> ExecResult:
        del timeout_s
        exit_code, output = self._container.exec_run(
            command,
            workdir=cwd,
            environment=env,
            user=user,
            demux=True,
        )
        stdout = ""
        stderr = ""
        if isinstance(output, tuple):
            stdout = _decode_output(output[0])
            stderr = _decode_output(output[1])
        else:
            stdout = _decode_output(output)
        rendered = command if isinstance(command, str) else " ".join(command)
        return ExecResult(command=rendered, exit_code=exit_code, stdout=stdout, stderr=stderr)

    def put_archive(self, remote_dir: str, data: bytes) -> None:
        if not self._container.put_archive(remote_dir, data):
            raise EnvironmentTransferError(f"docker.put_archive rejected path={remote_dir!r}")

    def get_archive(self, remote_path: str) -> bytes:
        stream, _stat = self._container.get_archive(remote_path)
        return b"".join(stream)

    def stop(self, *, delete: bool = True) -> None:
        self._container.stop()
        if delete:
            self._container.remove(force=True)

    def logs(self, *, stream: str | None = None) -> str:
        stdout = stream in {None, "stdout"}
        stderr = stream in {None, "stderr"}
        return _decode_output(self._container.logs(stdout=stdout, stderr=stderr))


def _sdk_volumes(mounts: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    volumes: dict[str, dict[str, str]] = {}
    for mount in mounts:
        volumes[mount["source"]] = {
            "bind": mount["target"],
            "mode": "ro" if mount.get("read_only") else "rw",
        }
    return volumes


__all__ = ["DockerEnvironment", "DockerEnvironmentProvider"]
