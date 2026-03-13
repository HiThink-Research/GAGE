import pytest

from gage_eval.sandbox.docker_runtime import DockerSandbox
from gage_eval.sandbox.local_runtime import LocalSubprocessSandbox
from gage_eval.sandbox.manager import SandboxManager
from gage_eval.sandbox.remote_runtime import RemoteSandbox


@pytest.mark.fast
def test_runtime_profile_registry():
    manager = SandboxManager()
    cases = [
        ("docker", DockerSandbox),
        ("local", LocalSubprocessSandbox),
        ("remote", RemoteSandbox),
        ("aio", DockerSandbox),
        ("appworld", DockerSandbox),
        ("llm", DockerSandbox),
        ("opensandbox", DockerSandbox),
    ]
    for runtime, expected_cls in cases:
        handle = manager.acquire({"runtime": runtime})
        assert isinstance(handle.sandbox, expected_cls)
        manager.release(handle)


@pytest.mark.fast
def test_deprecated_runtime_alias_class_still_exists() -> None:
    with pytest.warns(DeprecationWarning):
        from gage_eval.sandbox.aio_runtime import AioSandbox

    assert issubclass(AioSandbox, DockerSandbox)
