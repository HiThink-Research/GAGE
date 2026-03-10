"""Sandbox runtime package."""

from __future__ import annotations

from typing import Any

from gage_eval.sandbox.base import (
    BaseSandbox,
    ExecResult,
    SandboxOptionalMixin,
    serialize_exec_result,
)
from gage_eval.sandbox.docker_runtime import DockerSandbox
from gage_eval.sandbox.local_runtime import LocalSubprocessSandbox
from gage_eval.sandbox.manager import SandboxHandle, SandboxManager
from gage_eval.sandbox.pool import SandboxPool
from gage_eval.sandbox.protocols import (
    StateQueryProtocol,
    TaskInitProtocol,
    ToolExecutionProtocol,
)
from gage_eval.sandbox.provider import SandboxProvider, SandboxScope
from gage_eval.sandbox.remote_runtime import RemoteSandbox
from gage_eval.sandbox.tau2_runtime import Tau2Runtime

__all__ = [
    "BaseSandbox",
    "ExecResult",
    "SandboxOptionalMixin",
    "serialize_exec_result",
    "DockerSandbox",
    "LocalSubprocessSandbox",
    "RemoteSandbox",
    "AioSandbox",
    "AppWorldRuntime",
    "LlmSandbox",
    "OpenSandbox",
    "Tau2Runtime",
    "SandboxHandle",
    "SandboxManager",
    "SandboxPool",
    "SandboxProvider",
    "SandboxScope",
    "ToolExecutionProtocol",
    "StateQueryProtocol",
    "TaskInitProtocol",
]


def __getattr__(name: str) -> Any:
    deprecated_exports = {
        "AioSandbox": ("gage_eval.sandbox.aio_runtime", "AioSandbox"),
        "AppWorldRuntime": ("gage_eval.sandbox.appworld_runtime", "AppWorldRuntime"),
        "LlmSandbox": ("gage_eval.sandbox.llm_runtime", "LlmSandbox"),
        "OpenSandbox": ("gage_eval.sandbox.opensandbox_runtime", "OpenSandbox"),
    }
    target = deprecated_exports.get(name)
    if target is None:
        raise AttributeError(f"module 'gage_eval.sandbox' has no attribute {name!r}")
    module_path, attr_name = target
    module = __import__(module_path, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
