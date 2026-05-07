"""E2B-backed AgentKit v2 environment provider."""

from __future__ import annotations

import asyncio
import inspect
import posixpath
import shlex
import shutil
import time
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from gage_eval.environment.contracts import (
    DEFAULT_READ_FILE_LIMIT_BYTES,
    BaseEnvironment,
    ExecResult,
    FileInfo,
    truncate_streams_for_exec_result,
    validate_read_size,
)
from gage_eval.environment.errors import (
    EnvironmentAttachError,
    EnvironmentCreateError,
    EnvironmentError,
    EnvironmentExecError,
    EnvironmentFileNotFoundError,
    EnvironmentPreflightError,
    EnvironmentTimeoutError,
    EnvironmentTransferError,
)
from gage_eval.environment.profiles import EnvironmentProfile
from gage_eval.environment.resources import EnvironmentCapabilities, EnvironmentResources, NetworkPolicy

from .config import E2BEnvironmentConfig


class E2BEnvironmentProvider:
    """Registry-facing provider with an injectable E2B client adapter."""

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
        client = self._client or _default_e2b_client()
        network_policy = config.network_policy if config.network_policy is not None else resources.network_policy
        allow_internet_access = _allow_internet_access(network_policy)
        environment = {**config.startup_env, **startup_env}
        safe_metadata = {
            "profile_id": profile_id,
            **{key: str(value) for key, value in metadata.items() if isinstance(value, str)},
        }
        timeout = config.sandbox_timeout_s if config.sandbox_timeout_s is not None else resources.timeout_s
        create_kwargs: dict[str, Any] = {
            "template": config.template_id,
            "metadata": safe_metadata,
            "envs": environment,
            "allow_internet_access": allow_internet_access,
        }
        if timeout is not None:
            create_kwargs["timeout"] = timeout

        try:
            sandbox = await _maybe_to_thread(client.create_sandbox, **create_kwargs)
        except EnvironmentError:
            raise
        except Exception as exc:
            if _is_timeout_exception(exc):
                raise EnvironmentTimeoutError(f"e2b.create timeout template={config.template_id!r}") from exc
            raise EnvironmentCreateError(f"e2b.create template={config.template_id!r}: {exc.__class__.__name__}") from exc

        env_id = f"e2b-{uuid4().hex}"
        return E2BEnvironment(
            env_id=env_id,
            name=f"gage-e2b-{env_id[-12:]}",
            sandbox=sandbox,
            config=config,
            network_policy=network_policy,
            allow_internet_access=allow_internet_access,
            metadata=safe_metadata,
        )


class E2BEnvironment:
    env_id: str
    name: str
    provider = "e2b"

    def __init__(
        self,
        *,
        env_id: str,
        name: str,
        sandbox: Any,
        config: E2BEnvironmentConfig,
        network_policy: NetworkPolicy,
        allow_internet_access: bool,
        metadata: dict[str, str],
    ) -> None:
        self.env_id = env_id
        self.name = name
        self.metadata = metadata
        self._sandbox = sandbox
        self._config = config
        self._network_policy = network_policy
        self._allow_internet_access = allow_internet_access
        self._stopped = False
        self._last_stop_mode: str | None = None
        self._last_attach_mode: str | None = "fresh"
        self._stdout_log: list[str] = []
        self._stderr_log: list[str] = []
        self.capabilities = EnvironmentCapabilities(
            supports_mounts=False,
            supports_upload_download=True,
            supports_internet_control=True,
            supports_privileged_dind=False,
            default_user=config.user,
        )

    async def start(self, *, force_build: bool = False) -> None:
        del force_build
        self._ensure_active()

    async def attach(self) -> None:
        self._ensure_active()
        self._last_attach_mode = "fresh"

    async def stop(self, *, delete: bool = True) -> None:
        del delete
        if self._stopped:
            return
        try:
            kill = getattr(self._sandbox, "kill", None)
            if callable(kill):
                await _maybe_to_thread(kill)
        except Exception as exc:
            raise EnvironmentCreateError(f"e2b.stop env_id={self.env_id}: {exc.__class__.__name__}") from exc
        self._stopped = True
        self._last_stop_mode = "deleted"

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
        self._ensure_active()
        cmd = _e2b_command(command, shell=shell)
        run = self._command_runner()
        start = time.monotonic()
        try:
            raw_result = await _maybe_to_thread(
                run,
                cmd,
                envs=env,
                cwd=cwd,
                timeout=timeout_s,
                request_timeout=self._config.request_timeout_s,
                user=user if user is not None else self._config.user,
            )
        except EnvironmentError:
            raise
        except Exception as exc:
            if _is_command_exit_exception(exc):
                raw_result = exc
            elif _is_timeout_exception(exc):
                raise EnvironmentTimeoutError(f"e2b.exec timeout timeout_s={timeout_s} command={command!r}") from exc
            else:
                raise EnvironmentExecError(f"e2b.exec command={command!r}: {exc.__class__.__name__}") from exc

        result = _exec_result(raw_result, command=command, duration_s=time.monotonic() - start)
        if result.timed_out:
            raise EnvironmentTimeoutError(f"e2b.exec timeout timeout_s={timeout_s} command={command!r}")
        result = truncate_streams_for_exec_result(
            result,
            stdout_max_bytes=self._config.stdout_limit_bytes,
            stderr_max_bytes=self._config.stderr_limit_bytes,
        )
        self._stdout_log.append(result.stdout)
        self._stderr_log.append(result.stderr)
        return result

    async def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        source = Path(local_path)
        if not source.is_file():
            raise EnvironmentFileNotFoundError(f"local file not found: {source}")
        try:
            await self.write_file(remote_path, source.read_bytes())
        except EnvironmentError:
            raise
        except Exception as exc:
            raise _transfer_error("upload_file", remote_path, exc) from exc

    async def upload_dir(self, local_path: str | Path, remote_path: str) -> None:
        self._ensure_active()
        source = Path(local_path)
        if not source.is_dir():
            raise EnvironmentFileNotFoundError(f"local dir not found: {source}")
        try:
            await self._make_dir(remote_path)
            for directory in sorted((entry for entry in source.rglob("*") if entry.is_dir()), key=lambda path: str(path)):
                await self._make_dir(_join_remote(remote_path, directory.relative_to(source).as_posix()))
            for file_path in sorted((entry for entry in source.rglob("*") if entry.is_file()), key=lambda path: str(path)):
                await self.write_file(_join_remote(remote_path, file_path.relative_to(source).as_posix()), file_path.read_bytes())
        except EnvironmentError:
            raise
        except Exception as exc:
            raise _transfer_error("upload_dir", remote_path, exc) from exc

    async def download_file(self, remote_path: str, local_path: str | Path) -> None:
        target = Path(local_path)
        data = await self._read_file_unbounded(remote_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        except Exception as exc:
            raise _transfer_error("download_file", remote_path, exc) from exc

    async def download_dir(self, remote_path: str, local_path: str | Path) -> None:
        target = Path(local_path)
        if not await self.is_dir(remote_path):
            raise EnvironmentFileNotFoundError(f"remote dir not found: {remote_path}")
        temp_target = target.parent / f".{target.name}.tmp-{uuid4().hex}"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temp_target.mkdir()
            await self._download_dir_entries(remote_path, temp_target)
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            temp_target.rename(target)
        except EnvironmentError:
            shutil.rmtree(temp_target, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(temp_target, ignore_errors=True)
            raise _transfer_error("download_dir", remote_path, exc) from exc

    async def write_file(self, path: str, content: bytes | str) -> None:
        self._ensure_active()
        payload = content.encode("utf-8") if isinstance(content, str) else content
        try:
            await _maybe_to_thread(self._files().write, path, payload, **self._file_kwargs())
        except EnvironmentError:
            raise
        except Exception as exc:
            raise _transfer_error("write_file", path, exc) from exc

    async def read_file(self, path: str, *, max_bytes: int = DEFAULT_READ_FILE_LIMIT_BYTES) -> bytes:
        content = await self._read_file_unbounded(path)
        validate_read_size(path, len(content), max_bytes=max_bytes)
        return content

    async def list_files(self, path: str) -> list[FileInfo]:
        if not await self.is_dir(path):
            raise EnvironmentFileNotFoundError(f"remote dir not found: {path}")
        result = await self.exec(
            f"find {_quote_remote_path(path)} -maxdepth 1 -mindepth 1 -printf '%y %s %p\\n'",
            shell="sh",
        )
        if result.exit_code != 0:
            raise EnvironmentFileNotFoundError(f"remote dir not found: {path}")
        return [_file_info_from_find_line(line) for line in result.stdout.splitlines() if line.strip()]

    async def is_file(self, path: str) -> bool:
        result = await self.exec(f"test -f {_quote_remote_path(path)}", shell="sh")
        return result.exit_code == 0

    async def is_dir(self, path: str) -> bool:
        result = await self.exec(f"test -d {_quote_remote_path(path)}", shell="sh")
        return result.exit_code == 0

    async def _read_file_unbounded(self, path: str) -> bytes:
        self._ensure_active()
        try:
            data = await _maybe_to_thread(self._files().read, path, format="bytes", **self._file_kwargs())
        except FileNotFoundError as exc:
            raise EnvironmentFileNotFoundError(f"remote file not found: {path}") from exc
        except EnvironmentError:
            raise
        except Exception as exc:
            if _is_not_found_exception(exc):
                raise EnvironmentFileNotFoundError(f"remote file not found: {path}") from exc
            raise _transfer_error("read_file", path, exc) from exc
        content = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        return content

    async def get_logs(self, *, stream: Literal["stdout", "stderr"] | None = None) -> str:
        self._ensure_active()
        if stream == "stdout":
            return "".join(self._stdout_log)
        if stream == "stderr":
            return "".join(self._stderr_log)
        return "".join(self._stdout_log + self._stderr_log)

    async def describe(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "name": self.name,
            "provider": self.provider,
            "capabilities": self.capabilities.model_dump(mode="python"),
            "metadata": dict(self.metadata),
            "diagnostics": {
                "sandbox_id": _sandbox_id(self._sandbox),
                "template_id": self._config.template_id,
                "network_policy": self._network_policy,
                "allow_internet_access": self._allow_internet_access,
                "request_timeout_s": self._config.request_timeout_s,
                "sandbox_timeout_s": self._config.sandbox_timeout_s,
                "persistence_supported": False,
                "last_stop_mode": self._last_stop_mode,
                "last_attach_mode": self._last_attach_mode,
                "stdout_limit_bytes": self._config.stdout_limit_bytes,
                "stderr_limit_bytes": self._config.stderr_limit_bytes,
            },
        }

    async def _download_dir_entries(self, remote_path: str, target: Path) -> None:
        entries = await self.list_files(remote_path)
        for entry in entries:
            child_target = target / posixpath.basename(entry.path)
            if entry.kind == "dir":
                child_target.mkdir(parents=True, exist_ok=True)
                await self._download_dir_entries(entry.path, child_target)
                continue
            if entry.kind == "file":
                await self.download_file(entry.path, child_target)

    async def _make_dir(self, path: str) -> None:
        try:
            await _maybe_to_thread(self._files().make_dir, path, **self._file_kwargs())
        except AttributeError:
            result = await self.exec(f"mkdir -p {_quote_remote_path(path)}", shell="sh")
            if result.exit_code != 0:
                raise EnvironmentTransferError(f"e2b.mkdir path={path!r} exit_code={result.exit_code}")
        except EnvironmentError:
            raise
        except Exception as exc:
            raise _transfer_error("make_dir", path, exc) from exc

    def _command_runner(self) -> Any:
        commands = getattr(self._sandbox, "commands", None)
        run = getattr(commands, "run", None)
        if not callable(run):
            raise EnvironmentExecError(f"e2b.commands.run unavailable env_id={self.env_id}")
        return run

    def _files(self) -> Any:
        files = getattr(self._sandbox, "files", None)
        if files is None:
            raise EnvironmentTransferError(f"e2b.files unavailable env_id={self.env_id}")
        return files

    def _file_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self._config.user is not None:
            kwargs["user"] = self._config.user
        if self._config.request_timeout_s is not None:
            kwargs["request_timeout"] = self._config.request_timeout_s
        return kwargs

    def _ensure_active(self) -> None:
        if self._stopped or self._sandbox is None:
            raise EnvironmentAttachError(f"e2b.environment inactive env_id={self.env_id}")


def _coerce_config(provider_config: Any, *, profile: EnvironmentProfile) -> E2BEnvironmentConfig:
    if isinstance(provider_config, E2BEnvironmentConfig):
        return provider_config
    if isinstance(provider_config, BaseModel):
        raw = provider_config.model_dump(mode="python", exclude_none=True)
    elif isinstance(provider_config, dict):
        raw = dict(provider_config)
    else:
        raw = {}
    profile_defaults = {key: value for key, value in profile.config.items() if key in E2BEnvironmentConfig.model_fields}
    profile_defaults.update(raw)
    try:
        return E2BEnvironmentConfig.model_validate(profile_defaults)
    except ValidationError as exc:
        summary = _validation_error_summary(exc)
        raise EnvironmentPreflightError(f"e2b.config validation failed: {summary}") from None


def _validation_error_summary(exc: ValidationError) -> str:
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    details: list[str] = []
    for error in errors:
        loc_parts = tuple(_safe_validation_loc_part(str(part)) for part in error.get("loc", ()) if str(part))
        loc = ".".join(loc_parts) if loc_parts else "__root__"
        error_type = str(error.get("type") or "validation_error")
        message = _safe_validation_message(str(error.get("msg") or "validation failed"))
        details.append(f"{loc}:{error_type}:{message}")
    return "; ".join(details) if details else "validation_error"


def _safe_validation_loc_part(part: str) -> str:
    if _looks_sensitive_path_part(part):
        return "<redacted>"
    return part[:80]


def _safe_validation_message(message: str) -> str:
    if "/" in message or "\\" in message:
        return "validation failed"
    return message[:160]


def _looks_sensitive_path_part(value: str) -> bool:
    if "\x00" in value or "/" in value or "\\" in value:
        return True
    if value in {".", ".."}:
        return True
    return value.startswith(("./", "../", "~/"))


def _allow_internet_access(network_policy: NetworkPolicy) -> bool:
    if network_policy == "block":
        return False
    if network_policy in {"allow", "egress_only"}:
        return True
    raise EnvironmentPreflightError(f"e2b.network_policy unsupported={network_policy!r}")


def _e2b_command(command: str, *, shell: Literal["sh", "login", "none"]) -> str:
    if shell == "login":
        return f". /etc/profile >/dev/null 2>&1 || true; {command}"
    if shell in {"sh", "none"}:
        if not command:
            raise EnvironmentPreflightError("e2b.exec command must be non-empty")
        return command
    raise EnvironmentPreflightError(f"e2b.exec unsupported shell={shell!r}")


def _exec_result(result: Any, *, command: str, duration_s: float) -> ExecResult:
    if isinstance(result, ExecResult):
        update = {"command": command}
        if result.duration_s is None:
            update["duration_s"] = duration_s
        return result.model_copy(update=update)
    if isinstance(result, dict):
        metadata = dict(result.get("metadata") or {})
        if result.get("error"):
            metadata["error"] = _decode_output(result.get("error"))
        return ExecResult(
            command=command,
            exit_code=_first_present(result, "exit_code", "exitCode", "return_code", "returncode", "code"),
            stdout=_decode_output(result.get("stdout", "")),
            stderr=_decode_output(result.get("stderr", "")),
            duration_s=result.get("duration_s", duration_s),
            timed_out=bool(result.get("timed_out", result.get("timeout", False))),
            truncated=bool(result.get("truncated", False)),
            metadata=metadata,
        )
    if result is None:
        return ExecResult(command=command, exit_code=0, duration_s=duration_s)

    metadata: dict[str, Any] = {}
    error = _first_attr(result, "error")
    if error:
        metadata["error"] = _decode_output(error)
    return ExecResult(
        command=command,
        exit_code=_first_attr(result, "exit_code", "exitCode", "return_code", "returncode", "code"),
        stdout=_decode_output(_first_attr(result, "stdout", "out", default="")),
        stderr=_decode_output(_first_attr(result, "stderr", "err", default="")),
        duration_s=_first_attr(result, "duration_s", "duration", default=duration_s),
        timed_out=bool(_first_attr(result, "timed_out", "timeout", default=False)),
        truncated=bool(_first_attr(result, "truncated", default=False)),
        metadata=metadata,
    )


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _first_attr(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _decode_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _file_info_from_find_line(line: str) -> FileInfo:
    kind_code, size, path = line.split(" ", 2)
    kind = {"f": "file", "d": "dir", "l": "symlink"}.get(kind_code, "other")
    return FileInfo(path=path, kind=kind, size_bytes=int(size) if kind == "file" else None)


def _quote_remote_path(path: str) -> str:
    if not isinstance(path, str) or not path:
        raise EnvironmentTransferError(f"remote path must be non-empty: {path!r}")
    if "\x00" in path:
        raise EnvironmentTransferError(f"remote path contains NUL byte: {path!r}")
    return shlex.quote(path)


def _join_remote(root: str, relative: str) -> str:
    if not relative or relative == ".":
        return root
    return posixpath.join(root.rstrip("/") or "/", relative)


def _transfer_error(operation: str, remote_path: str, exc: BaseException) -> EnvironmentTransferError:
    if isinstance(exc, EnvironmentTransferError):
        return exc
    detail = exc.__class__.__name__
    errno = getattr(exc, "errno", None)
    if errno is not None:
        detail = f"{detail} errno={errno}"
    return EnvironmentTransferError(f"e2b.{operation} path={remote_path!r} failed: {detail}")


def _sandbox_id(sandbox: Any) -> str | None:
    value = getattr(sandbox, "sandbox_id", None) or getattr(sandbox, "id", None)
    return str(value) if value is not None else None


def _is_timeout_exception(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    return "timeout" in name or "timed out" in message


def _is_command_exit_exception(exc: BaseException) -> bool:
    if any(cls.__name__ == "CommandExitException" for cls in type(exc).mro()):
        return True
    return all(hasattr(exc, name) for name in ("exit_code", "stdout", "stderr"))


def _is_not_found_exception(exc: BaseException) -> bool:
    return isinstance(exc, FileNotFoundError) or any("notfound" in cls.__name__.lower() for cls in type(exc).mro())


async def _maybe_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    result = await asyncio.to_thread(func, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _default_e2b_client() -> Any:
    return _E2BSDKClientAdapter()


class _E2BSDKClientAdapter:
    def create_sandbox(self, **kwargs: Any) -> Any:
        try:
            from e2b import Sandbox
        except ImportError as exc:
            raise EnvironmentCreateError("e2b.sdk missing; install the e2b Python package") from exc
        return Sandbox.create(**kwargs)


__all__ = ["E2BEnvironment", "E2BEnvironmentProvider"]
