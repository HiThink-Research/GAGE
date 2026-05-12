from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Literal, get_args, get_type_hints

import pytest
from pydantic import ValidationError

from gage_eval.environment import (
    DEFAULT_EXEC_STREAM_LIMIT_BYTES,
    DEFAULT_READ_FILE_LIMIT_BYTES,
    BaseEnvironment,
    EnvironmentAttachError,
    EnvironmentCapabilities,
    EnvironmentCreateError,
    EnvironmentError,
    EnvironmentExecError,
    EnvironmentFileNotFoundError,
    EnvironmentPreflightError,
    EnvironmentResources,
    EnvironmentTimeoutError,
    EnvironmentTransferError,
    EnvironmentFileConvenienceMixin,
    ExecResult,
    FileInfo,
    ensure_environment_error,
    truncate_streams_for_exec_result,
    validate_phase1_persistence_descriptor,
    validate_read_size,
)


@pytest.mark.fast
def test_exec_result_schema_contains_expected_fields_and_forbids_extras() -> None:
    result = ExecResult(
        command="python -V",
        exit_code=0,
        stdout="Python 3.12\n",
        stderr="",
        duration_s=0.25,
        output_artifact_refs=[{"artifact_id": "stdout-log"}],
        metadata={"provider": "local_process"},
    )

    assert result.command == "python -V"
    assert result.exit_code == 0
    assert result.stdout == "Python 3.12\n"
    assert result.stderr == ""
    assert result.duration_s == 0.25
    assert result.timed_out is False
    assert result.truncated is False
    assert result.output_artifact_refs == [{"artifact_id": "stdout-log"}]
    assert result.metadata == {"provider": "local_process"}

    with pytest.raises(ValidationError):
        ExecResult(command="cmd", exit_code=0, stdout="", stderr="", duration_s=0, extra=True)


@pytest.mark.fast
def test_exec_result_stdout_stderr_truncation_is_explicit_and_defaults_to_4_mib() -> None:
    result = ExecResult(command="cmd", exit_code=0, stdout="abcdef", stderr="uvwxyz", duration_s=0.1)

    truncated = truncate_streams_for_exec_result(
        result,
        stdout_max_bytes=4,
        stderr_max_bytes=3,
    )

    assert DEFAULT_EXEC_STREAM_LIMIT_BYTES == 4 * 1024 * 1024
    assert truncated.stdout == "abcd"
    assert truncated.stderr == "uvw"
    assert truncated.truncated is True
    assert truncated.command == "cmd"
    assert truncated.exit_code == 0
    assert truncated.output_artifact_refs == []


@pytest.mark.fast
def test_file_info_and_environment_resources_are_strict_public_records() -> None:
    file_info = FileInfo(path="/workspace/file.txt", kind="file", size_bytes=12)
    resources = EnvironmentResources(cpu=2.0, memory_gb=4.0, disk_gb=20.0, timeout_s=60)
    capabilities = EnvironmentCapabilities(default_user="agent")

    assert file_info.kind == "file"
    assert set(get_args(FileInfo.model_fields["kind"].annotation)) == {"file", "dir", "symlink", "other"}
    assert resources.network_policy == "block"
    assert capabilities.supports_mounts is False
    assert capabilities.supports_upload_download is True
    assert capabilities.supports_internet_control is False
    assert capabilities.supports_privileged_dind is False

    with pytest.raises(ValidationError):
        EnvironmentResources(network_policy="open")
    with pytest.raises(ValidationError):
        FileInfo(path="/tmp/x", kind="file", unexpected=True)


@pytest.mark.fast
def test_public_records_reject_pydantic_type_coercion() -> None:
    with pytest.raises(ValidationError):
        EnvironmentResources(cpu="2.0")
    with pytest.raises(ValidationError):
        EnvironmentCapabilities(supports_mounts="true")
    with pytest.raises(ValidationError):
        FileInfo(path="/tmp/x", kind="file", size_bytes="12")
    with pytest.raises(ValidationError):
        ExecResult(command="cmd", exit_code="0")


@pytest.mark.fast
def test_public_records_reject_impossible_negative_values() -> None:
    with pytest.raises(ValidationError):
        ExecResult(command="cmd", exit_code=0, duration_s=-0.1)
    with pytest.raises(ValidationError):
        FileInfo(path="/tmp/x", kind="file", size_bytes=-1)
    with pytest.raises(ValidationError):
        EnvironmentResources(cpu=-1)
    with pytest.raises(ValidationError):
        EnvironmentResources(memory_gb=-2)
    with pytest.raises(ValidationError):
        EnvironmentResources(disk_gb=-3)
    with pytest.raises(ValidationError):
        EnvironmentResources(timeout_s=-4)


@pytest.mark.fast
def test_base_environment_protocol_exposes_benchmark_neutral_async_surface() -> None:
    expected_attrs = {"env_id", "name", "provider", "metadata", "capabilities"}
    expected_methods = {
        "start",
        "attach",
        "stop",
        "exec",
        "upload_file",
        "upload_dir",
        "download_file",
        "download_dir",
        "write_file",
        "read_file",
        "list_files",
        "is_file",
        "is_dir",
        "get_logs",
        "describe",
    }
    alias_methods = {"upload", "download", "write", "read", "list"}

    assert expected_attrs <= set(BaseEnvironment.__annotations__)
    for method_name in expected_methods:
        assert hasattr(BaseEnvironment, method_name), method_name
        assert inspect.iscoroutinefunction(getattr(BaseEnvironment, method_name)), method_name
    for method_name in alias_methods:
        assert not hasattr(BaseEnvironment, method_name), method_name


@pytest.mark.fast
def test_canonical_only_environment_satisfies_base_environment_protocol() -> None:
    class CanonicalOnlyEnvironment:
        env_id = "env"
        name = "demo"
        provider = "fake"
        metadata: dict[str, str] = {}
        capabilities = EnvironmentCapabilities()

        async def start(self, *, force_build: bool = False) -> None:
            pass

        async def attach(self) -> None:
            pass

        async def stop(self, *, delete: bool = True) -> None:
            pass

        async def exec(
            self,
            command: str,
            *,
            env: dict[str, str] | None = None,
            cwd: str | None = None,
            timeout_s: int | None = None,
            user: str | None = None,
            shell: Literal["sh", "login", "none"] = "sh",
        ) -> ExecResult:
            return ExecResult(command=command, exit_code=0)

        async def upload_file(self, local_path: str | Path, remote_path: str) -> None:
            pass

        async def upload_dir(self, local_path: str | Path, remote_path: str) -> None:
            pass

        async def download_file(self, remote_path: str, local_path: str | Path) -> None:
            pass

        async def download_dir(self, remote_path: str, local_path: str | Path) -> None:
            pass

        async def write_file(self, path: str, content: bytes | str) -> None:
            pass

        async def read_file(self, path: str, *, max_bytes: int = DEFAULT_READ_FILE_LIMIT_BYTES) -> bytes:
            return b""

        async def list_files(self, path: str) -> list[FileInfo]:
            return []

        async def is_file(self, path: str) -> bool:
            return False

        async def is_dir(self, path: str) -> bool:
            return False

        async def get_logs(self, *, stream: Literal["stdout", "stderr"] | None = None) -> str:
            return ""

        async def describe(self) -> dict[str, object]:
            return {}

    fake = CanonicalOnlyEnvironment()

    assert isinstance(fake, BaseEnvironment)
    for method_name in ("upload", "download", "write", "read", "list"):
        assert not hasattr(fake, method_name), method_name


@pytest.mark.fast
def test_file_convenience_mixin_aliases_delegate_to_canonical_methods() -> None:
    class ConvenienceEnvironment(EnvironmentFileConvenienceMixin):
        def __init__(self) -> None:
            self.calls: list[tuple[str, object, object | None]] = []

        async def upload_file(self, local_path: str | Path, remote_path: str) -> None:
            self.calls.append(("upload_file", local_path, remote_path))

        async def download_file(self, remote_path: str, local_path: str | Path) -> None:
            self.calls.append(("download_file", remote_path, local_path))

        async def write_file(self, path: str, content: bytes | str) -> None:
            self.calls.append(("write_file", path, content))

        async def read_file(self, path: str, *, max_bytes: int = DEFAULT_READ_FILE_LIMIT_BYTES) -> bytes:
            self.calls.append(("read_file", path, max_bytes))
            return b"payload"

        async def list_files(self, path: str) -> list[FileInfo]:
            self.calls.append(("list_files", path, None))
            return [FileInfo(path=f"{path}/x", kind="file")]

    async def exercise() -> ConvenienceEnvironment:
        fake = ConvenienceEnvironment()
        await fake.upload("/tmp/local.txt", "/remote.txt")
        await fake.download("/remote.txt", "/tmp/local.txt")
        await fake.write("/remote.txt", b"content")
        assert await fake.read("/remote.txt", max_bytes=8) == b"payload"
        assert await fake.list("/remote") == [FileInfo(path="/remote/x", kind="file")]
        return fake

    fake = asyncio.run(exercise())

    assert fake.calls == [
        ("upload_file", "/tmp/local.txt", "/remote.txt"),
        ("download_file", "/remote.txt", "/tmp/local.txt"),
        ("write_file", "/remote.txt", b"content"),
        ("read_file", "/remote.txt", 8),
        ("list_files", "/remote", None),
    ]


@pytest.mark.fast
def test_base_environment_protocol_signatures_match_design() -> None:
    protocol_hints = get_type_hints(BaseEnvironment)
    assert protocol_hints["metadata"] == dict[str, str]

    start_signature = inspect.signature(BaseEnvironment.start)
    assert list(start_signature.parameters) == ["self", "force_build"]
    assert start_signature.parameters["force_build"].kind is inspect.Parameter.KEYWORD_ONLY
    assert start_signature.parameters["force_build"].default is False
    start_hints = get_type_hints(BaseEnvironment.start)
    assert start_hints == {"force_build": bool, "return": type(None)}

    attach_signature = inspect.signature(BaseEnvironment.attach)
    assert list(attach_signature.parameters) == ["self"]
    attach_hints = get_type_hints(BaseEnvironment.attach)
    assert attach_hints == {"return": type(None)}

    exec_signature = inspect.signature(BaseEnvironment.exec)
    assert list(exec_signature.parameters) == ["self", "command", "env", "cwd", "timeout_s", "user", "shell"]
    assert exec_signature.parameters["command"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for keyword_name in ("env", "cwd", "timeout_s", "user", "shell"):
        assert exec_signature.parameters[keyword_name].kind is inspect.Parameter.KEYWORD_ONLY
    assert exec_signature.parameters["env"].default is None
    assert exec_signature.parameters["cwd"].default is None
    assert exec_signature.parameters["timeout_s"].default is None
    assert exec_signature.parameters["user"].default is None
    assert exec_signature.parameters["shell"].default == "sh"
    assert "stdout_max_bytes" not in exec_signature.parameters
    assert "stderr_max_bytes" not in exec_signature.parameters
    exec_hints = get_type_hints(BaseEnvironment.exec)
    assert exec_hints == {
        "command": str,
        "env": dict[str, str] | None,
        "cwd": str | None,
        "timeout_s": int | None,
        "user": str | None,
        "shell": Literal["sh", "login", "none"],
        "return": ExecResult,
    }

    logs_signature = inspect.signature(BaseEnvironment.get_logs)
    assert list(logs_signature.parameters) == ["self", "stream"]
    assert logs_signature.parameters["stream"].kind is inspect.Parameter.KEYWORD_ONLY
    assert logs_signature.parameters["stream"].default is None
    logs_hints = get_type_hints(BaseEnvironment.get_logs)
    assert logs_hints == {"stream": Literal["stdout", "stderr"] | None, "return": str}


@pytest.mark.fast
def test_base_environment_canonical_file_method_parameter_names_match_design() -> None:
    expected_parameters = {
        "upload_file": ["self", "local_path", "remote_path"],
        "upload_dir": ["self", "local_path", "remote_path"],
        "download_file": ["self", "remote_path", "local_path"],
        "download_dir": ["self", "remote_path", "local_path"],
        "write_file": ["self", "path", "content"],
    }

    for method_name, parameter_names in expected_parameters.items():
        signature = inspect.signature(getattr(BaseEnvironment, method_name))
        assert list(signature.parameters) == parameter_names


@pytest.mark.fast
def test_base_environment_protocol_excludes_tau2_specific_methods() -> None:
    tau2_specific_methods = {
        "mark_agent_exhausted",
        "configure_user_simulator",
        "initialize_task",
        "get_state",
        "exec_tool",
        "step_user_simulator",
    }

    for method_name in tau2_specific_methods:
        assert not hasattr(BaseEnvironment, method_name), method_name


@pytest.mark.fast
def test_read_file_small_file_limit_and_error_family_are_testable() -> None:
    validate_read_size("/workspace/small.txt", DEFAULT_READ_FILE_LIMIT_BYTES)

    with pytest.raises(EnvironmentPreflightError, match="read_file.max_bytes"):
        validate_read_size("/workspace/small.txt", 1, max_bytes=-1)
    with pytest.raises(EnvironmentPreflightError, match="read_file.size_bytes"):
        validate_read_size("/workspace/invalid.txt", -1)
    with pytest.raises(EnvironmentTransferError, match="read_file.size_limit"):
        validate_read_size("/workspace/large.txt", DEFAULT_READ_FILE_LIMIT_BYTES + 1)

    for error_cls in (
        EnvironmentPreflightError,
        EnvironmentCreateError,
        EnvironmentAttachError,
        EnvironmentExecError,
        EnvironmentFileNotFoundError,
        EnvironmentTransferError,
        EnvironmentTimeoutError,
    ):
        assert issubclass(error_cls, EnvironmentError)
        assert ensure_environment_error(error_cls("provider failure")).__class__ is error_cls

    with pytest.raises(TypeError, match="EnvironmentError"):
        ensure_environment_error(RuntimeError("raw provider error"))


@pytest.mark.fast
@pytest.mark.parametrize(
    "descriptor",
    [
        {},
        {"last_stop_mode": None, "last_attach_mode": None},
        {"last_stop_mode": "deleted", "last_attach_mode": "fresh"},
        {"last_stop_mode": "unsupported", "last_attach_mode": None},
    ],
)
def test_describe_persistence_fields_phase1_constraints_accept_supported_values(
    descriptor: dict[str, object],
) -> None:
    assert validate_phase1_persistence_descriptor(descriptor) == descriptor


@pytest.mark.fast
@pytest.mark.parametrize(
    "descriptor",
    [
        {"last_stop_mode": "paused"},
        {"last_stop_mode": "stopped"},
        {"last_stop_mode": "resumed"},
        {"last_attach_mode": "attached"},
        {"last_attach_mode": "resumed"},
    ],
)
def test_describe_persistence_fields_phase1_constraints_reject_future_values(
    descriptor: dict[str, object],
) -> None:
    with pytest.raises(EnvironmentPreflightError, match="environment.persistence.phase1"):
        validate_phase1_persistence_descriptor(descriptor)


@pytest.mark.fast
def test_requirements_declares_pydantic_v2_baseline() -> None:
    requirements = (Path(__file__).resolve().parents[3] / "requirements.txt").read_text(encoding="utf-8")

    assert "pydantic>=2.0.0" in requirements
