"""Benchmark-neutral environment protocol for AgentKit v2."""

from gage_eval.environment.contracts import (
    DEFAULT_EXEC_STREAM_LIMIT_BYTES,
    DEFAULT_READ_FILE_LIMIT_BYTES,
    BaseEnvironment,
    EnvironmentFileConvenienceMixin,
    ExecResult,
    FileInfo,
    truncate_streams_for_exec_result,
    validate_phase1_persistence_descriptor,
    validate_read_size,
)
from gage_eval.environment.lease import EnvironmentLease
from gage_eval.environment.manager import (
    DEFAULT_RETRY_BUDGET_BY_FAILURE,
    EnvironmentManager,
    EnvironmentManagerError,
)
from gage_eval.environment.profiles import EnvironmentProfile
from gage_eval.environment.providers.registry import EnvironmentProvider, ProviderRegistry
from gage_eval.environment.errors import (
    EnvironmentAttachError,
    EnvironmentCreateError,
    EnvironmentError,
    EnvironmentExecError,
    EnvironmentFileNotFoundError,
    EnvironmentPreflightError,
    EnvironmentTimeoutError,
    EnvironmentTransferError,
    ensure_environment_error,
)
from gage_eval.environment.resources import EnvironmentCapabilities, EnvironmentResources

__all__ = [
    "DEFAULT_EXEC_STREAM_LIMIT_BYTES",
    "DEFAULT_READ_FILE_LIMIT_BYTES",
    "BaseEnvironment",
    "EnvironmentAttachError",
    "EnvironmentCapabilities",
    "EnvironmentCreateError",
    "EnvironmentError",
    "EnvironmentExecError",
    "EnvironmentFileConvenienceMixin",
    "EnvironmentFileNotFoundError",
    "EnvironmentLease",
    "EnvironmentManager",
    "EnvironmentManagerError",
    "EnvironmentProfile",
    "EnvironmentPreflightError",
    "EnvironmentProvider",
    "EnvironmentResources",
    "EnvironmentTimeoutError",
    "EnvironmentTransferError",
    "ExecResult",
    "FileInfo",
    "ProviderRegistry",
    "DEFAULT_RETRY_BUDGET_BY_FAILURE",
    "ensure_environment_error",
    "truncate_streams_for_exec_result",
    "validate_phase1_persistence_descriptor",
    "validate_read_size",
]
