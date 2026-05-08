"""Error types and validation helpers for benchmark-neutral environments."""

from __future__ import annotations


class EnvironmentError(Exception):
    """Base class for all environment provider failures."""


class EnvironmentPreflightError(EnvironmentError):
    """Raised when an environment request fails validation before provider work."""


class EnvironmentCreateError(EnvironmentError):
    """Raised when a provider cannot create a new environment."""


class EnvironmentAttachError(EnvironmentError):
    """Raised when a provider cannot attach to an existing environment."""


class EnvironmentExecError(EnvironmentError):
    """Raised when command execution fails outside normal process exit status."""


class EnvironmentFileNotFoundError(EnvironmentError):
    """Raised when a requested environment path does not exist."""


class EnvironmentTransferError(EnvironmentError):
    """Raised when upload, download, read, or write operations fail."""


class EnvironmentTimeoutError(EnvironmentError):
    """Raised when an environment operation exceeds its timeout."""


def ensure_environment_error(exc: BaseException) -> EnvironmentError:
    """Reject provider exceptions that are not part of the environment family."""

    if isinstance(exc, EnvironmentError):
        return exc
    raise TypeError(f"provider exceptions must inherit EnvironmentError: {exc!r}")
