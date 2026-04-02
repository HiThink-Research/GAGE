from __future__ import annotations

import pytest

from gage_eval.sandbox.contracts import RemoteSandboxContract, RemoteSandboxHandle
from gage_eval.sandbox.surfaces import build_remote_surfaces


@pytest.mark.fast
def test_build_remote_surfaces_marks_fs_partial_when_only_exec_exists() -> None:
    contract = RemoteSandboxContract(
        mode="attached",
        exec_endpoint="http://remote/api/run_command",
    )
    handle = RemoteSandboxHandle(
        mode="attached",
        exec_url="http://remote/api/run_command",
    )

    surfaces = build_remote_surfaces(contract, handle)

    assert surfaces["terminal"].status == "available"
    assert surfaces["fs"].status == "partial"
    assert surfaces["fs"].reason == "using_terminal_fallback"
