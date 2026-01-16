import pytest

from gage_eval.sandbox.aio_runtime import AioSandbox
from gage_eval.sandbox.appworld_runtime import AppWorldRuntime
from gage_eval.sandbox.docker_runtime import DockerSandbox
from gage_eval.sandbox.llm_runtime import LlmSandbox
from gage_eval.sandbox.local_runtime import LocalSubprocessSandbox
from gage_eval.sandbox.manager import SandboxManager
from gage_eval.sandbox.opensandbox_runtime import OpenSandbox
from gage_eval.sandbox.remote_runtime import RemoteSandbox


@pytest.mark.fast
def test_runtime_profile_registry():
    manager = SandboxManager()
    cases = [
        ("docker", DockerSandbox),
        ("local", LocalSubprocessSandbox),
        ("remote", RemoteSandbox),
        ("aio", AioSandbox),
        ("appworld", AppWorldRuntime),
        ("llm", LlmSandbox),
        ("opensandbox", OpenSandbox),
    ]
    for runtime, expected_cls in cases:
        handle = manager.acquire({"runtime": runtime})
        assert isinstance(handle.sandbox, expected_cls)
        manager.release(handle)
