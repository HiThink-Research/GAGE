"""OpenSandbox runtime profile (Docker-based)."""

from __future__ import annotations

import warnings

from gage_eval.sandbox.docker_runtime import DockerSandbox


warnings.warn(
    "OpenSandbox is deprecated. Use DockerSandbox with sandbox_profiles instead.",
    DeprecationWarning,
    stacklevel=2,
)


class OpenSandbox(DockerSandbox):
    """Deprecated Docker-based OpenSandbox runtime alias."""
