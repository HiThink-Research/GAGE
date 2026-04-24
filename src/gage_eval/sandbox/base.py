"""Sandbox base interfaces and execution results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ExecResult:
    """Represents a single sandbox execution result."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the execution result to a JSON-friendly dict."""

        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
        }


class BaseSandbox:
    """Base interface for sandbox runtimes."""

    def start(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:  # pragma: no cover - interface
        """Start a sandbox runtime with the provided configuration."""

        raise NotImplementedError

    def exec(
        self,
        command: str,
        timeout: int = 30,
        *,
        login_shell: bool = True,
    ) -> ExecResult:  # pragma: no cover - interface
        """Execute a command in the sandbox runtime."""

        raise NotImplementedError

    def teardown(self) -> None:  # pragma: no cover - interface
        """Tear down and release sandbox resources."""

        raise NotImplementedError

    @classmethod
    def cleanup_stale_runtime(
        cls,
        config: Dict[str, Any],
        runtime_handle: Dict[str, Any],
    ) -> bool:  # pragma: no cover - optional
        """Clean up stale runtime resources from a previous crashed process.

        Args:
            config: Persisted sandbox configuration snapshot.
            runtime_handle: Persisted runtime handle snapshot.

        Returns:
            True when no further retry is needed for this lease.
        """

        return True


class SandboxOptionalMixin:
    """Optional sandbox capabilities (health checks, renewals, file I/O)."""

    def is_alive(self, timeout_s: float | None = None) -> bool:  # pragma: no cover - optional
        """Return whether the runtime is still healthy."""

        return True

    def describe_runtime_state(self, timeout_s: float | None = None) -> Dict[str, Any]:  # pragma: no cover - optional
        """Return a JSON-friendly runtime state snapshot for diagnostics."""

        return {}

    def renew(self, ttl_s: int | None = None) -> None:  # pragma: no cover - optional
        """Renew or extend the sandbox runtime."""

        return None

    def read_file(self, path: str) -> bytes:  # pragma: no cover - optional
        """Read a file from inside the sandbox runtime."""

        raise NotImplementedError

    def write_file(self, path: str, content: bytes) -> None:  # pragma: no cover - optional
        """Write a file into the sandbox runtime."""

        raise NotImplementedError


def serialize_exec_result(result: ExecResult | Dict[str, Any]) -> Dict[str, Any]:
    """Normalize execution results to a dict."""

    if isinstance(result, ExecResult):
        return result.to_dict()
    if isinstance(result, dict):
        return dict(result)
    return {"output": str(result)}
