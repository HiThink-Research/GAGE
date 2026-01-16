"""AppWorld runtime profile (Docker-based)."""

from __future__ import annotations

from gage_eval.sandbox.docker_runtime import DockerSandbox


class AppWorldRuntime(DockerSandbox):
    """Docker-based AppWorld sandbox runtime."""
