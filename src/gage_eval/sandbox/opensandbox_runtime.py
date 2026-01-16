"""OpenSandbox runtime profile (Docker-based)."""

from __future__ import annotations

from gage_eval.sandbox.docker_runtime import DockerSandbox


class OpenSandbox(DockerSandbox):
    """Docker-based OpenSandbox runtime."""
