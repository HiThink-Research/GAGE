"""AppWorld runtime profile (Docker-based)."""

from __future__ import annotations

import warnings

from gage_eval.sandbox.docker_runtime import DockerSandbox


warnings.warn(
    "AppWorldRuntime is deprecated. Use DockerSandbox with sandbox_profiles instead.",
    DeprecationWarning,
    stacklevel=2,
)


class AppWorldRuntime(DockerSandbox):
    """Deprecated Docker-based AppWorld sandbox runtime alias."""
