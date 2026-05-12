"""Host-local subprocess AgentKit v2 environment provider."""

from __future__ import annotations

import asyncio
import os
import posixpath
import shlex
import shutil
import signal
import tempfile
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
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
    EnvironmentExecError,
    EnvironmentFileNotFoundError,
    EnvironmentPreflightError,
    EnvironmentTimeoutError,
    EnvironmentTransferError,
)
from gage_eval.environment.profiles import EnvironmentProfile
from gage_eval.environment.resources import EnvironmentCapabilities, EnvironmentResources

from .config import LocalProcessEnvironmentConfig


POST_KILL_WAIT_TIMEOUT_S = 0.5


class LocalProcessEnvironmentProvider:
    """Registry-facing provider that runs commands as local subprocesses."""

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
        del kit_id, provider, resources, lifecycle
        config = _coerce_config(provider_config, profile=profile)
        env_id = f"local-process-{uuid4().hex}"
        try:
            workdir, owns_workdir = _create_workdir(config)
        except Exception as exc:
            raise EnvironmentCreateError(f"local_process.create workdir failed: {exc.__class__.__name__}") from exc

        environment = {**config.startup_env, **startup_env}
        return LocalProcessEnvironment(
            env_id=env_id,
            name=f"gage-local-process-{env_id[-12:]}",
            workdir=workdir,
            owns_workdir=owns_workdir,
            config=config,
            startup_env=environment,
            metadata={
                "profile_id": profile_id,
                **{key: str(value) for key, value in metadata.items() if isinstance(value, str)},
            },
        )


class LocalProcessEnvironment:
    env_id: str
    name: str
    provider = "local_process"

    def __init__(
        self,
        *,
        env_id: str,
        name: str,
        workdir: Path,
        owns_workdir: bool,
        config: LocalProcessEnvironmentConfig,
        startup_env: dict[str, str],
        metadata: dict[str, str],
    ) -> None:
        self.env_id = env_id
        self.name = name
        self.metadata = metadata
        self._workdir = workdir.resolve(strict=False)
        self._owns_workdir = owns_workdir
        self._config = config
        self._startup_env = startup_env
        self._stopped = False
        self._stdout_log: list[str] = []
        self._stderr_log: list[str] = []
        self.capabilities = EnvironmentCapabilities(
            supports_mounts=False,
            supports_upload_download=True,
            supports_internet_control=False,
            supports_privileged_dind=False,
            default_user=None,
        )

    async def start(self, *, force_build: bool = False) -> None:
        del force_build
        try:
            self._workdir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise EnvironmentCreateError(f"local_process.start workdir failed: {exc.__class__.__name__}") from exc
        self._stopped = False

    async def attach(self) -> None:
        if self._stopped or not self._workdir.is_dir():
            raise EnvironmentAttachError(f"local_process.attach env_id={self.env_id}")

    async def stop(self, *, delete: bool = True) -> None:
        if delete and self._owns_workdir:
            try:
                shutil.rmtree(self._workdir, ignore_errors=True)
            except Exception as exc:
                raise EnvironmentCreateError(f"local_process.stop failed: {exc.__class__.__name__}") from exc
        self._stopped = True

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
        if user is not None:
            raise EnvironmentPreflightError("local_process.exec user is unsupported")
        if timeout_s is not None and timeout_s < 0:
            raise EnvironmentPreflightError(f"local_process.exec timeout_s={timeout_s}")

        cwd_path = self._resolve_cwd(cwd)
        argv = _exec_argv(command, shell=shell)
        process_env = os.environ.copy()
        process_env.update(self._startup_env)
        if env is not None:
            process_env.update(env)

        start = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(cwd_path),
                env=process_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError as exc:
                await _kill_process_tree(process)
                raise EnvironmentTimeoutError(
                    f"local_process.exec timeout timeout_s={timeout_s} command={command!r}"
                ) from exc
        except EnvironmentTimeoutError:
            raise
        except EnvironmentPreflightError:
            raise
        except Exception as exc:
            raise EnvironmentExecError(f"local_process.exec command={command!r}: {exc.__class__.__name__}") from exc

        result = ExecResult(
            command=command,
            exit_code=process.returncode,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            duration_s=time.monotonic() - start,
        )
        result = truncate_streams_for_exec_result(
            result,
            stdout_max_bytes=self._config.stdout_limit_bytes,
            stderr_max_bytes=self._config.stderr_limit_bytes,
        )
        self._stdout_log.append(result.stdout)
        self._stderr_log.append(result.stderr)
        return result

    async def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        self._ensure_active()
        source = Path(local_path)
        if not source.is_file():
            raise EnvironmentFileNotFoundError("local file not found")
        target = self._resolve_transfer_path(remote_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        except Exception as exc:
            raise _transfer_error("upload_file", remote_path, exc) from exc

    async def upload_dir(self, local_path: str | Path, remote_path: str) -> None:
        self._ensure_active()
        source = Path(local_path)
        if not source.is_dir():
            raise EnvironmentFileNotFoundError("local dir not found")
        target = self._resolve_transfer_path(remote_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not target.is_dir():
                    raise EnvironmentTransferError(f"local_process.upload_dir target is not a dir: {remote_path!r}")
                shutil.rmtree(target)
            shutil.copytree(source, target)
        except EnvironmentTransferError:
            raise
        except Exception as exc:
            raise _transfer_error("upload_dir", remote_path, exc) from exc

    async def download_file(self, remote_path: str, local_path: str | Path) -> None:
        self._ensure_active()
        source = self._resolve_transfer_path(remote_path)
        if not source.is_file():
            raise EnvironmentFileNotFoundError(f"remote file not found: {remote_path}")
        target = Path(local_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        except Exception as exc:
            raise _transfer_error("download_file", remote_path, exc) from exc

    async def download_dir(self, remote_path: str, local_path: str | Path) -> None:
        self._ensure_active()
        source = self._resolve_transfer_path(remote_path)
        if not source.is_dir():
            raise EnvironmentFileNotFoundError(f"remote dir not found: {remote_path}")
        target = Path(local_path)
        try:
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.copytree(source, target)
        except Exception as exc:
            raise _transfer_error("download_dir", remote_path, exc) from exc

    async def write_file(self, path: str, content: bytes | str) -> None:
        self._ensure_active()
        payload = content.encode("utf-8") if isinstance(content, str) else content
        host_path = self._resolve_transfer_path(path)
        try:
            host_path.parent.mkdir(parents=True, exist_ok=True)
            host_path.write_bytes(payload)
        except Exception as exc:
            raise _transfer_error("write_file", path, exc) from exc

    async def read_file(self, path: str, *, max_bytes: int = DEFAULT_READ_FILE_LIMIT_BYTES) -> bytes:
        self._ensure_active()
        host_path = self._resolve_transfer_path(path)
        if not host_path.is_file():
            raise EnvironmentFileNotFoundError(f"remote file not found: {path}")
        validate_read_size(path, host_path.stat().st_size, max_bytes=max_bytes)
        try:
            return host_path.read_bytes()
        except Exception as exc:
            raise _transfer_error("read_file", path, exc) from exc

    async def list_files(self, path: str) -> list[FileInfo]:
        self._ensure_active()
        host_path = self._resolve_transfer_path(path)
        if not host_path.is_dir():
            raise EnvironmentFileNotFoundError(f"remote dir not found: {path}")
        remote_root = _remote_display(path)
        try:
            entries = sorted(host_path.iterdir(), key=lambda entry: entry.name)
            return [_file_info_from_path(entry, posixpath.join(remote_root, entry.name)) for entry in entries]
        except Exception as exc:
            raise _transfer_error("list_files", path, exc) from exc

    async def is_file(self, path: str) -> bool:
        self._ensure_active()
        return self._resolve_transfer_path(path).is_file()

    async def is_dir(self, path: str) -> bool:
        self._ensure_active()
        return self._resolve_transfer_path(path).is_dir()

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
                "strong_isolation": False,
                "isolation": "local_process_not_strongly_isolated",
                "warning": "local_process runs on the host and does not provide strong security isolation",
                "workdir": "provider-managed" if self._owns_workdir else "configured",
                "stdout_limit_bytes": self._config.stdout_limit_bytes,
                "stderr_limit_bytes": self._config.stderr_limit_bytes,
            },
        }

    def _ensure_active(self) -> None:
        if self._stopped or not self._workdir.is_dir():
            raise EnvironmentAttachError(f"local_process.environment inactive env_id={self.env_id}")

    def _resolve_cwd(self, cwd: str | None) -> Path:
        if cwd is None:
            return self._workdir
        try:
            host_path = self._resolve_path(cwd, error_cls=EnvironmentPreflightError)
        except EnvironmentPreflightError:
            raise
        if not host_path.is_dir():
            raise EnvironmentPreflightError(f"local_process.exec cwd not found: {cwd!r}")
        return host_path

    def _resolve_transfer_path(self, path: str) -> Path:
        return self._resolve_path(path, error_cls=EnvironmentTransferError)

    def _resolve_path(
        self,
        path: str,
        *,
        error_cls: type[EnvironmentPreflightError] | type[EnvironmentTransferError],
    ) -> Path:
        relative = _safe_relative_path(path, error_cls=error_cls)
        self._reject_symlink_chain(relative, original_path=path, error_cls=error_cls)
        if relative == PurePosixPath("."):
            return self._workdir
        host_path = self._workdir.joinpath(*relative.parts)
        host_resolved = host_path.resolve(strict=False)
        parent_resolved = host_path.parent.resolve(strict=False)
        if not _path_is_relative_to(host_resolved, self._workdir) or not _path_is_relative_to(
            parent_resolved,
            self._workdir,
        ):
            raise error_cls(f"local_process.path escapes workdir: {path!r}")
        return host_path

    def _reject_symlink_chain(
        self,
        relative: PurePosixPath,
        *,
        original_path: str,
        error_cls: type[EnvironmentPreflightError] | type[EnvironmentTransferError],
    ) -> None:
        current = self._workdir
        if current.is_symlink():
            raise error_cls(f"local_process.path contains symlink: {original_path!r}")
        for part in relative.parts:
            if part in {"", "."}:
                continue
            current = current / part
            if (current.exists() or current.is_symlink()) and current.is_symlink():
                raise error_cls(f"local_process.path contains symlink: {original_path!r}")


def _create_workdir(config: LocalProcessEnvironmentConfig) -> tuple[Path, bool]:
    if config.workdir is not None:
        workdir = Path(config.workdir)
        if workdir.exists() and workdir.is_symlink():
            raise EnvironmentCreateError("configured workdir must not be a symlink")
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir.resolve(strict=False), False
    if config.base_cwd is not None:
        base_cwd = Path(config.base_cwd)
        if base_cwd.exists() and base_cwd.is_symlink():
            raise EnvironmentCreateError("base_cwd must not be a symlink")
        base_cwd.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="gage-local-process-", dir=base_cwd)).resolve(strict=False), True
    return Path(tempfile.mkdtemp(prefix="gage-local-process-")).resolve(strict=False), True


def _coerce_config(provider_config: Any, *, profile: EnvironmentProfile) -> LocalProcessEnvironmentConfig:
    if isinstance(provider_config, LocalProcessEnvironmentConfig):
        return provider_config
    if isinstance(provider_config, BaseModel):
        raw = provider_config.model_dump(mode="python", exclude_none=True)
    elif isinstance(provider_config, dict):
        raw = dict(provider_config)
    else:
        raw = {}
    profile_defaults = {
        key: value for key, value in profile.config.items() if key in LocalProcessEnvironmentConfig.model_fields
    }
    profile_defaults.update(raw)
    try:
        return LocalProcessEnvironmentConfig.model_validate(profile_defaults)
    except ValidationError as exc:
        summary = _validation_error_summary(exc)
        raise EnvironmentPreflightError(f"local_process.config validation failed: {summary}") from None


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


def _exec_argv(command: str, *, shell: Literal["sh", "login", "none"]) -> list[str]:
    if shell == "sh":
        return ["/bin/sh", "-lc", command]
    if shell == "login":
        return ["/bin/sh", "-lc", f". /etc/profile >/dev/null 2>&1 || true; {command}"]
    if shell == "none":
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise EnvironmentPreflightError(f"local_process.exec command parse failed: {exc}") from exc
        if not argv:
            raise EnvironmentPreflightError("local_process.exec command must be non-empty")
        return argv
    raise EnvironmentPreflightError(f"local_process.exec unsupported shell={shell!r}")


async def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    descendant_pids = await _collect_descendant_pids(process.pid)
    _kill_pids(reversed(descendant_pids))
    if process.returncode is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await _wait_for_process_exit(process)
    _kill_pids(reversed(descendant_pids))


async def _collect_descendant_pids(root_pid: int) -> list[int]:
    try:
        process = await asyncio.create_subprocess_exec(
            "ps",
            "-eo",
            "pid=,ppid=",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=POST_KILL_WAIT_TIMEOUT_S)
    except Exception:
        return []

    children_by_parent: dict[int, list[int]] = {}
    for line in stdout.decode("utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        children_by_parent.setdefault(ppid, []).append(pid)

    descendants: list[int] = []
    pending = list(children_by_parent.get(root_pid, []))
    while pending:
        pid = pending.pop()
        descendants.append(pid)
        pending.extend(children_by_parent.get(pid, []))
    return descendants


def _kill_pids(pids: Any) -> None:
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


async def _wait_for_process_exit(process: asyncio.subprocess.Process) -> None:
    try:
        await asyncio.wait_for(process.wait(), timeout=POST_KILL_WAIT_TIMEOUT_S)
    except asyncio.TimeoutError:
        return


def _safe_relative_path(
    path: str,
    *,
    error_cls: type[EnvironmentPreflightError] | type[EnvironmentTransferError],
) -> PurePosixPath:
    if not isinstance(path, str) or not path:
        raise error_cls(f"local_process.path must be non-empty: {path!r}")
    if "\x00" in path:
        raise error_cls(f"local_process.path contains NUL byte: {path!r}")
    parsed = PurePosixPath(path)
    if any(part == ".." for part in parsed.parts):
        raise error_cls(f"local_process.path contains '..': {path!r}")
    normalized = posixpath.normpath(path)
    if normalized == ".":
        return PurePosixPath(".")
    if normalized == "/":
        return PurePosixPath(".")
    if normalized.startswith("/"):
        normalized = normalized.lstrip("/")
    return PurePosixPath(normalized)


def _remote_display(path: str) -> str:
    relative = _safe_relative_path(path, error_cls=EnvironmentTransferError)
    if relative == PurePosixPath("."):
        return "/"
    return f"/{relative.as_posix()}"


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _file_info_from_path(path: Path, remote_path: str) -> FileInfo:
    try:
        stat_result = path.lstat()
    except FileNotFoundError:
        raise EnvironmentFileNotFoundError(f"remote path not found: {remote_path}") from None
    if path.is_symlink():
        kind = "symlink"
        size_bytes = None
    elif path.is_file():
        kind = "file"
        size_bytes = stat_result.st_size
    elif path.is_dir():
        kind = "dir"
        size_bytes = None
    else:
        kind = "other"
        size_bytes = None
    return FileInfo(
        path=remote_path,
        kind=kind,
        size_bytes=size_bytes,
        modified_at=datetime.fromtimestamp(stat_result.st_mtime),
    )


def _transfer_error(operation: str, remote_path: str, exc: BaseException) -> EnvironmentTransferError:
    if isinstance(exc, EnvironmentTransferError):
        return exc
    detail = exc.__class__.__name__
    errno = getattr(exc, "errno", None)
    if errno is not None:
        detail = f"{detail} errno={errno}"
    return EnvironmentTransferError(f"local_process.{operation} path={remote_path!r} failed: {detail}")


__all__ = ["LocalProcessEnvironment", "LocalProcessEnvironmentProvider"]
