from __future__ import annotations

import pytest

from gage_eval.sandbox.contracts import (
    RemoteSandboxContract,
    coerce_remote_sandbox_handle,
    merge_sandbox_profile_layers,
)


@pytest.mark.fast
def test_remote_contract_requires_control_endpoint_for_managed() -> None:
    with pytest.raises(ValueError):
        RemoteSandboxContract(mode="managed")


@pytest.mark.fast
def test_merge_sandbox_profile_layers_respects_profile_defaults() -> None:
    merged = merge_sandbox_profile_layers(
        {"remote-codex": {"exec_endpoint": "http://profile/exec", "auth": {"headers": {"x-profile": "1"}}}},
        "remote-codex",
        {"mode": "attached"},
        {"auth": {"headers": {"x-local": "1"}}},
    )

    assert merged["exec_endpoint"] == "http://profile/exec"
    assert merged["auth"]["headers"]["x-profile"] == "1"
    assert merged["auth"]["headers"]["x-local"] == "1"


@pytest.mark.fast
def test_coerce_remote_handle_normalizes_remote_mode() -> None:
    handle = coerce_remote_sandbox_handle(
        {
            "remote_mode": "attached",
            "exec_url": "http://remote/exec",
            "surface_names": ["terminal", "fs"],
        }
    )

    assert handle.mode == "attached"
    assert handle.surface_names == ("terminal", "fs")
