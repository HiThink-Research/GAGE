"""Remote environment wrapper — shell only, not yet implemented."""

from __future__ import annotations


class RemoteEnvironment:
    """Placeholder remote-backed AgentEnvironment implementation."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def start(self) -> dict:
        raise NotImplementedError("RemoteEnvironment.start is not implemented yet")

    def stop(self) -> None:
        raise NotImplementedError("RemoteEnvironment.stop is not implemented yet")

    def exec(self, command: str, *, cwd=None, env=None, timeout_sec: int = 30):
        raise NotImplementedError("RemoteEnvironment.exec is not implemented yet")

    def upload_file(self, local_path: str, remote_path: str) -> None:
        raise NotImplementedError("RemoteEnvironment.upload_file is not implemented yet")

    def download_file(self, remote_path: str, local_path: str) -> None:
        raise NotImplementedError("RemoteEnvironment.download_file is not implemented yet")

    def read_file(self, remote_path: str) -> bytes:
        raise NotImplementedError("RemoteEnvironment.read_file is not implemented yet")

    def write_file(self, remote_path: str, content: bytes) -> None:
        raise NotImplementedError("RemoteEnvironment.write_file is not implemented yet")
