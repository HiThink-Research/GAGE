"""Sandbox policy helpers — no extra dataclass definitions here."""

from __future__ import annotations

from dataclasses import replace

from gage_eval.sandbox.contracts import RemoteSandboxContract, validate_remote_sandbox_contract
from gage_eval.agent_runtime.spec import SandboxPolicy


def resolve_sandbox_policy(base: SandboxPolicy, sample: dict) -> SandboxPolicy:
    """Merge runtime-level and sample-level sandbox policy."""

    sample_policy = sample.get("sandbox_policy") if isinstance(sample, dict) else None
    if not isinstance(sample_policy, dict):
        return base
    params = dict(base.params)
    params.update(sample_policy.get("params") or {})
    return replace(
        base,
        sandbox_profile_id=sample_policy.get("sandbox_profile_id", base.sandbox_profile_id),
        prefer_remote=sample_policy.get("prefer_remote", base.prefer_remote),
        allow_local_fallback=sample_policy.get("allow_local_fallback", base.allow_local_fallback),
        remote_mode=sample_policy.get("remote_mode", base.remote_mode),
        params=params,
    )


def validate_remote_contract(contract: RemoteSandboxContract) -> None:
    """Validate attached and managed endpoint invariants."""

    validate_remote_sandbox_contract(contract)
