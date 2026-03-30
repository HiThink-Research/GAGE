"""Docker environment wrapper — shell only, not yet implemented."""

from __future__ import annotations


class DockerEnvironment:
    """Placeholder Docker-backed AgentEnvironment implementation."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def start(self) -> dict:
        raise NotImplementedError("DockerEnvironment.start is not implemented yet")

    def stop(self) -> None:
        raise NotImplementedError("DockerEnvironment.stop is not implemented yet")

    def exec(self, command: str, *, cwd=None, env=None, timeout_sec: int = 30):
        raise NotImplementedError("DockerEnvironment.exec is not implemented yet")

    def upload_file(self, local_path: str, remote_path: str) -> None:
        raise NotImplementedError("DockerEnvironment.upload_file is not implemented yet")

    def download_file(self, remote_path: str, local_path: str) -> None:
        raise NotImplementedError("DockerEnvironment.download_file is not implemented yet")

    def read_file(self, remote_path: str) -> bytes:
        raise NotImplementedError("DockerEnvironment.read_file is not implemented yet")

    def write_file(self, remote_path: str, content: bytes) -> None:
        raise NotImplementedError("DockerEnvironment.write_file is not implemented yet")
