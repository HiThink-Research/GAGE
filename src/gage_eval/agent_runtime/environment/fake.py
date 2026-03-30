"""Fake environment for unit testing — no real sandbox needed."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from gage_eval.sandbox.base import ExecResult


class FakeEnvironment:
    """In-memory fake that records all operations for assertions."""

    def __init__(self) -> None:
        self.started = False
        self.commands: List[str] = []
        self.files: Dict[str, bytes] = {}
        self.exec_results: Dict[str, ExecResult] = {}

    def start(self) -> dict:
        self.started = True
        return {"fake": True}

    def stop(self) -> None:
        self.started = False

    def exec(
        self,
        command: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        timeout_sec: int = 30,
    ) -> ExecResult:
        self.commands.append(command)
        if command in self.exec_results:
            return self.exec_results[command]
        return ExecResult(exit_code=0, stdout="", stderr="")

    def upload_file(self, local_path: str, remote_path: str) -> None:
        self.files[remote_path] = b"<uploaded>"

    def download_file(self, remote_path: str, local_path: str) -> None:
        return None

    def read_file(self, remote_path: str) -> bytes:
        return self.files.get(remote_path, b"")

    def write_file(self, remote_path: str, content: bytes) -> None:
        self.files[remote_path] = content
