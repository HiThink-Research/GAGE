from __future__ import annotations

import pytest

from gage_eval.agent_runtime.environment.fake import FakeEnvironment
from gage_eval.agent_runtime.resources.bundle import ResourceBundle
from gage_eval.agent_runtime.resources.client_surface import ClientSurface, build_remote_surfaces
from gage_eval.agent_runtime.resources.remote_sandbox import RemoteSandboxContract, RemoteSandboxHandle


@pytest.mark.fast
def test_remote_sandbox_attached_requires_exec_endpoint() -> None:
    with pytest.raises(ValueError):
        RemoteSandboxContract(mode="attached")


@pytest.mark.fast
def test_remote_sandbox_managed_requires_control_endpoint() -> None:
    with pytest.raises(ValueError):
        RemoteSandboxContract(mode="managed")


@pytest.mark.fast
def test_client_surface_defaults() -> None:
    surface = ClientSurface(surface_type="terminal")

    assert surface.status == "available"


@pytest.mark.fast
def test_build_remote_surfaces_derives_terminal_fs_and_mcp() -> None:
    contract = RemoteSandboxContract(
        mode="attached",
        exec_endpoint="http://remote/api/run_command",
        file_endpoint="http://remote/api",
        mcp_endpoint="http://remote/mcp",
    )
    handle = RemoteSandboxHandle(
        mode="attached",
        exec_url="http://remote/api/run_command",
        data_endpoint="http://remote/api",
        mcp_endpoint="http://remote/mcp",
    )

    surfaces = build_remote_surfaces(contract, handle)

    assert tuple(surfaces) == ("terminal", "fs", "mcp")
    assert surfaces["terminal"].endpoint == "http://remote/api/run_command"
    assert surfaces["fs"].endpoint == "http://remote/api"
    assert surfaces["mcp"].endpoint == "http://remote/mcp"


@pytest.mark.fast
def test_resource_bundle_require_surface_missing() -> None:
    bundle = ResourceBundle(environment=FakeEnvironment())

    with pytest.raises(KeyError):
        bundle.require_surface("terminal")


@pytest.mark.fast
def test_resource_bundle_get_surface_returns_none() -> None:
    bundle = ResourceBundle(environment=FakeEnvironment())

    assert bundle.get_surface("terminal") is None
