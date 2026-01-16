"""LLM sandbox runtime profile (Docker-based)."""

from __future__ import annotations

from gage_eval.sandbox.docker_runtime import DockerSandbox


class LlmSandbox(DockerSandbox):
    """Docker-based LLM sandbox runtime."""
