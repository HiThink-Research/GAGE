"""Benchmark-neutral environment protocol for AgentKit v2."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from gage_eval.environment.errors import EnvironmentPreflightError, EnvironmentTransferError
from gage_eval.environment.resources import EnvironmentCapabilities

DEFAULT_EXEC_STREAM_LIMIT_BYTES = 4 * 1024 * 1024
DEFAULT_READ_FILE_LIMIT_BYTES = 16 * 1024 * 1024

FileKind = Literal["file", "dir", "symlink", "other"]
StopModePhase1 = Literal["deleted", "unsupported"]
AttachModePhase1 = Literal["fresh"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=(), strict=True)


class ExecResult(_StrictModel):
    command: str
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    duration_s: float | None = Field(default=None, ge=0)
    timed_out: bool = False
    truncated: bool = False
    output_artifact_refs: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FileInfo(_StrictModel):
    path: str
    kind: FileKind
    size_bytes: int | None = Field(default=None, ge=0)
    modified_at: datetime | None = None


def truncate_streams_for_exec_result(
    result: ExecResult,
    *,
    stdout_max_bytes: int = DEFAULT_EXEC_STREAM_LIMIT_BYTES,
    stderr_max_bytes: int = DEFAULT_EXEC_STREAM_LIMIT_BYTES,
) -> ExecResult:
    """Return an ExecResult with stdout/stderr capped by UTF-8 byte length."""

    stdout, stdout_truncated = _truncate_text_by_utf8(result.stdout, stdout_max_bytes)
    stderr, stderr_truncated = _truncate_text_by_utf8(result.stderr, stderr_max_bytes)
    return result.model_copy(
        update={
            "stdout": stdout,
            "stderr": stderr,
            "truncated": result.truncated or stdout_truncated or stderr_truncated,
        }
    )


def validate_read_size(
    path: str,
    size_bytes: int | None,
    *,
    max_bytes: int = DEFAULT_READ_FILE_LIMIT_BYTES,
) -> None:
    """Validate the default small-file read limit before returning file bytes."""

    if max_bytes < 0:
        raise EnvironmentPreflightError(f"read_file.max_bytes path={path!r} max_bytes={max_bytes}")
    if size_bytes is not None and size_bytes < 0:
        raise EnvironmentPreflightError(f"read_file.size_bytes path={path!r} size_bytes={size_bytes}")
    if size_bytes is not None and size_bytes > max_bytes:
        raise EnvironmentTransferError(
            f"read_file.size_limit path={path!r} size_bytes={size_bytes} max_bytes={max_bytes}"
        )


def validate_phase1_persistence_descriptor(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Fail fast on persistence modes reserved for later AgentKit phases."""

    last_stop_mode = descriptor.get("last_stop_mode")
    if last_stop_mode not in {"deleted", "unsupported", None}:
        raise EnvironmentPreflightError(
            f"environment.persistence.phase1.unsupported last_stop_mode={last_stop_mode!r}"
        )

    last_attach_mode = descriptor.get("last_attach_mode")
    if last_attach_mode not in {"fresh", None}:
        raise EnvironmentPreflightError(
            f"environment.persistence.phase1.unsupported last_attach_mode={last_attach_mode!r}"
        )

    return dict(descriptor)


def _truncate_text_by_utf8(text: str, max_bytes: int) -> tuple[str, bool]:
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


@runtime_checkable
class BaseEnvironment(Protocol):
    env_id: str
    name: str
    provider: str
    metadata: dict[str, str]
    capabilities: EnvironmentCapabilities

    async def start(self, *, force_build: bool = False) -> None:
        ...

    async def attach(self) -> None:
        ...

    async def stop(self, *, delete: bool = True) -> None:
        ...

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
        ...

    async def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        ...

    async def upload_dir(self, local_path: str | Path, remote_path: str) -> None:
        ...

    async def download_file(self, remote_path: str, local_path: str | Path) -> None:
        ...

    async def download_dir(self, remote_path: str, local_path: str | Path) -> None:
        ...

    async def write_file(self, path: str, content: bytes | str) -> None:
        ...

    async def read_file(
        self,
        path: str,
        *,
        max_bytes: int = DEFAULT_READ_FILE_LIMIT_BYTES,
    ) -> bytes:
        ...

    async def list_files(self, path: str) -> list[FileInfo]:
        ...

    async def is_file(self, path: str) -> bool:
        ...

    async def is_dir(self, path: str) -> bool:
        ...

    async def get_logs(self, *, stream: Literal["stdout", "stderr"] | None = None) -> str:
        ...

    async def describe(self) -> dict[str, Any]:
        ...


class EnvironmentFileConvenienceMixin:
    """Optional aliases implemented in terms of canonical file methods."""

    async def upload(self, local_path: str | Path, remote_path: str) -> None:
        await self.upload_file(local_path, remote_path)

    async def download(self, remote_path: str, local_path: str | Path) -> None:
        await self.download_file(remote_path, local_path)

    async def write(self, path: str, content: bytes | str) -> None:
        await self.write_file(path, content)

    async def read(self, path: str, *, max_bytes: int = DEFAULT_READ_FILE_LIMIT_BYTES) -> bytes:
        return await self.read_file(path, max_bytes=max_bytes)

    async def list(self, path: str) -> list[FileInfo]:
        return await self.list_files(path)
