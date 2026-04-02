"""Compatibility re-export for the formal remote sandbox contracts."""

from __future__ import annotations

from gage_eval.sandbox.contracts import (
    RemoteSandboxContract,
    RemoteSandboxHandle,
    validate_remote_sandbox_contract,
)

__all__ = [
    "RemoteSandboxContract",
    "RemoteSandboxHandle",
    "validate_remote_sandbox_contract",
]
