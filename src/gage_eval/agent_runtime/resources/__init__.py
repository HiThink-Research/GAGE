from __future__ import annotations

from typing import Any

__all__ = [
    "ResourceLease",
    "RuntimeLeaseBinding",
    "RuntimeResourceManager",
]


def __getattr__(name: str) -> Any:
    """Lazily expose resource runtime symbols."""

    if name == "ResourceLease":
        from gage_eval.agent_runtime.resources.contracts import ResourceLease

        return ResourceLease
    if name in {"RuntimeLeaseBinding", "RuntimeResourceManager"}:
        from gage_eval.agent_runtime.resources.manager import (
            RuntimeLeaseBinding,
            RuntimeResourceManager,
        )

        return {
            "RuntimeLeaseBinding": RuntimeLeaseBinding,
            "RuntimeResourceManager": RuntimeResourceManager,
        }[name]
    raise AttributeError(name)
