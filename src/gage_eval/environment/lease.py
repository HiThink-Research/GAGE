"""Environment lease and runtime proxy helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from gage_eval.environment.contracts import (
    DEFAULT_EXEC_STREAM_LIMIT_BYTES,
    DEFAULT_READ_FILE_LIMIT_BYTES,
    BaseEnvironment,
    ExecResult,
    truncate_streams_for_exec_result,
)


@dataclass
class EnvironmentLease:
    """Exclusive lease for one acquired benchmark-neutral environment."""

    lease_id: str
    environment: BaseEnvironment
    provider: str
    profile_id: str
    lifecycle: Literal["per_sample", "per_task"]
    exclusive: bool
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_sink: Any | None = field(default=None, repr=False, compare=False)
    stdout_limit_bytes: int = DEFAULT_EXEC_STREAM_LIMIT_BYTES
    stderr_limit_bytes: int = DEFAULT_EXEC_STREAM_LIMIT_BYTES

    def to_descriptor(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "provider": self.provider,
            "profile_id": self.profile_id,
            "lifecycle": self.lifecycle,
            "exclusive": self.exclusive,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
            "environment_descriptor": self._environment_descriptor(),
        }

    def describe(self) -> dict[str, Any]:
        return self.to_descriptor()

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
        result = await self.environment.exec(
            command,
            env=env,
            cwd=cwd,
            timeout_s=timeout_s,
            user=user,
            shell=shell,
        )
        output_refs = list(result.output_artifact_refs or [])
        if self.artifact_sink is not None:
            output_refs.extend(await self._write_truncated_stream_artifacts(result))
        truncated = truncate_streams_for_exec_result(
            result,
            stdout_max_bytes=self.stdout_limit_bytes,
            stderr_max_bytes=self.stderr_limit_bytes,
        )
        return truncated.model_copy(update={"output_artifact_refs": output_refs})

    async def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        await self.environment.upload_file(local_path, remote_path)

    async def upload_dir(self, local_path: str | Path, remote_path: str) -> None:
        await self.environment.upload_dir(local_path, remote_path)

    async def download_file(self, remote_path: str, local_path: str | Path) -> None:
        await self.environment.download_file(remote_path, local_path)

    async def download_dir(self, remote_path: str, local_path: str | Path) -> None:
        await self.environment.download_dir(remote_path, local_path)

    async def write_file(self, path: str, content: bytes | str) -> None:
        await self.environment.write_file(path, content)

    async def read_file(self, path: str, *, max_bytes: int = DEFAULT_READ_FILE_LIMIT_BYTES) -> bytes:
        return await self.environment.read_file(path, max_bytes=max_bytes)

    async def list_files(self, path: str) -> list[Any]:
        return await self.environment.list_files(path)

    async def is_file(self, path: str) -> bool:
        return await self.environment.is_file(path)

    async def is_dir(self, path: str) -> bool:
        return await self.environment.is_dir(path)

    async def get_logs(self, *, stream: Literal["stdout", "stderr"] | None = None) -> str:
        return await self.environment.get_logs(stream=stream)

    async def exec_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        executor = getattr(self.environment, "exec_tool", None)
        if not callable(executor):
            executor = getattr(self.environment, "call_tool", None)
        if not callable(executor):
            raise AttributeError("environment does not expose exec_tool or call_tool")
        result = executor(name, arguments)
        if inspect.isawaitable(result):
            return await result
        return result

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self.exec_tool(name, arguments)

    async def _write_truncated_stream_artifacts(self, result: ExecResult) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        for stream, content, limit in (
            ("stdout", result.stdout, self.stdout_limit_bytes),
            ("stderr", result.stderr, self.stderr_limit_bytes),
        ):
            if len(content.encode("utf-8")) <= limit:
                continue
            ref = await self._write_artifact(
                owner=self.lease_id,
                name=f"{stream}.txt",
                content=content,
                metadata={
                    "stream": stream,
                    "command": result.command,
                    "env_id": getattr(self.environment, "env_id", None),
                    "truncated_limit_bytes": limit,
                },
            )
            refs.append({"stream": stream, **(ref if isinstance(ref, dict) else {"artifact_ref": ref})})
        return refs

    async def _write_artifact(
        self,
        *,
        owner: str,
        name: str,
        content: str,
        metadata: dict[str, Any],
    ) -> Any:
        writer = getattr(self.artifact_sink, "write_artifact")
        value = writer(owner=owner, name=name, content=content, metadata=metadata)
        if inspect.isawaitable(value):
            return await value
        return value

    def _environment_descriptor(self) -> dict[str, Any]:
        capabilities = getattr(self.environment, "capabilities", None)
        if hasattr(capabilities, "model_dump"):
            capabilities_payload = capabilities.model_dump(mode="python")
        elif isinstance(capabilities, dict):
            capabilities_payload = dict(capabilities)
        else:
            capabilities_payload = {}
        return {
            "env_id": getattr(self.environment, "env_id", None),
            "name": getattr(self.environment, "name", None),
            "provider": getattr(self.environment, "provider", self.provider),
            "capabilities": capabilities_payload,
            "metadata": dict(getattr(self.environment, "metadata", {}) or {}),
        }
