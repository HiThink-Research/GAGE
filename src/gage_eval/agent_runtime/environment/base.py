"""AgentEnvironment protocol — unified execution workspace contract."""

from __future__ import annotations

from typing import Mapping, Optional, Protocol

from gage_eval.sandbox.base import ExecResult


class AgentEnvironment(Protocol):
    """Uniform contract for local, docker, and remote execution workspaces."""

    def start(self) -> dict:
        """Start the environment."""

    def stop(self) -> None:
        """Stop and release environment resources."""

    def exec(
        self,
        command: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        timeout_sec: int = 30,
    ) -> ExecResult:
        """Execute a command and return an ExecResult."""

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a file into the environment."""

    def download_file(self, remote_path: str, local_path: str) -> None:
        """Download a file from the environment."""

    def read_file(self, remote_path: str) -> bytes:
        """Read a file inside the environment."""

    def write_file(self, remote_path: str, content: bytes) -> None:
        """Write a file inside the environment."""
