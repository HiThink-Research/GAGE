from __future__ import annotations

import asyncio
import re
import shlex
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from gage_eval.environment import (
    BaseEnvironment,
    DEFAULT_READ_FILE_LIMIT_BYTES,
    EnvironmentCapabilities,
    EnvironmentCreateError,
    EnvironmentFileNotFoundError,
    EnvironmentManager,
    EnvironmentManagerError,
    EnvironmentPreflightError,
    EnvironmentResources,
    EnvironmentTimeoutError,
    EnvironmentTransferError,
    ExecResult,
)
from gage_eval.environment.profiles import EnvironmentProfile
from gage_eval.environment.providers.e2b import E2BEnvironmentConfig, E2BEnvironmentProvider
from gage_eval.environment.providers.registry import create_default_provider_registry


class FakeE2BClient:
    def __init__(self, sandbox: "FakeSandbox | None" = None) -> None:
        self.sandbox = sandbox or FakeSandbox()
        self.create_calls: list[dict[str, Any]] = []

    def create_sandbox(self, **kwargs: Any) -> "FakeSandbox":
        self.create_calls.append(kwargs)
        return self.sandbox


class FakeSandbox:
    id = "sbx_123"
    sandbox_id = "sbx_123"

    def __init__(self) -> None:
        self.commands = FakeCommands()
        self.files = FakeFiles()
        self.kill_calls = 0

    def kill(self) -> None:
        self.kill_calls += 1


class FakeCommands:
    def __init__(self) -> None:
        self.run_calls: list[dict[str, Any]] = []
        self.results: list[Any] = []

    def run(self, cmd: Any, **kwargs: Any) -> Any:
        self.run_calls.append({"cmd": cmd, **kwargs})
        if self.results:
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return ExecResult(command=str(cmd), exit_code=0, stdout="ok\n")


class PathAwareCommands(FakeCommands):
    def run(self, cmd: Any, **kwargs: Any) -> Any:
        self.run_calls.append({"cmd": cmd, **kwargs})
        command = str(cmd)
        if command.startswith("test -d /workspace/file.txt"):
            return ExecResult(command=command, exit_code=1)
        if command.startswith("find /workspace/file.txt "):
            return ExecResult(command=command, exit_code=0, stdout="")
        if command.startswith("test -d /workspace/missing"):
            return ExecResult(command=command, exit_code=1)
        if command.startswith("find /workspace/missing "):
            return ExecResult(command=command, exit_code=1, stderr="missing")
        return ExecResult(command=command, exit_code=0, stdout="")


class FakeFiles:
    def __init__(self) -> None:
        self.storage: dict[str, bytes] = {}
        self.write_calls: list[dict[str, Any]] = []
        self.read_calls: list[dict[str, Any]] = []
        self.make_dir_calls: list[dict[str, Any]] = []

    def write(self, path: str, data: bytes | str, **kwargs: Any) -> None:
        self.write_calls.append({"path": path, "data": data, **kwargs})
        self.storage[path] = data.encode("utf-8") if isinstance(data, str) else data

    def read(self, path: str, **kwargs: Any) -> bytes:
        self.read_calls.append({"path": path, **kwargs})
        if path not in self.storage:
            raise FileNotFoundError(path)
        return self.storage[path]

    def make_dir(self, path: str, **kwargs: Any) -> None:
        self.make_dir_calls.append({"path": path, **kwargs})


class ObjectCommandResult:
    def __init__(
        self,
        *,
        exit_code: int | None = 0,
        stdout: bytes | str = "",
        stderr: bytes | str = "",
        error: str | None = None,
        timed_out: bool = False,
    ) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.error = error
        self.timed_out = timed_out


class CommandExitException(Exception):
    def __init__(
        self,
        *,
        exit_code: int,
        stdout: bytes | str = "",
        stderr: bytes | str = "",
        error: str | None = None,
    ) -> None:
        rendered_stderr = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr
        message = error or f"command exited with code {exit_code}"
        if rendered_stderr:
            message = f"{message}: {rendered_stderr}"
        super().__init__(message)
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.error = error


def _profile(config: dict[str, Any] | None = None) -> EnvironmentProfile:
    return EnvironmentProfile(profile_id="e2b-profile", provider="e2b", config=config or {})


def _resources(network_policy: str = "block") -> EnvironmentResources:
    return EnvironmentResources(cpu=1.0, memory_gb=2.0, timeout_s=30, network_policy=network_policy)  # type: ignore[arg-type]


async def _create_environment(
    provider: E2BEnvironmentProvider,
    config: E2BEnvironmentConfig | dict[str, Any],
    *,
    profile: EnvironmentProfile | None = None,
    resources: EnvironmentResources | None = None,
    startup_env: dict[str, str] | None = None,
) -> BaseEnvironment:
    return await provider.create(
        kit_id="tau2",
        provider="e2b",
        profile_id="e2b-profile",
        profile=profile or _profile(),
        provider_config=config,
        resources=resources or _resources(),
        startup_env=startup_env or {"RUN_LEVEL": "request"},
        lifecycle="per_sample",
        metadata={"sample_id": "sample-1"},
    )


@pytest.mark.fast
def test_e2b_config_schema_validates_template_network_env_user_and_limits() -> None:
    config = E2BEnvironmentConfig(
        template_id="base-python",
        sandbox_timeout_s=60,
        request_timeout_s=5,
        network_policy="egress_only",
        startup_env={"CONFIG_LEVEL": "1"},
        user="agent",
        stdout_limit_bytes=16,
        stderr_limit_bytes=8,
    )

    assert config.template_id == "base-python"
    assert config.network_policy == "egress_only"
    assert config.startup_env == {"CONFIG_LEVEL": "1"}
    assert config.user == "agent"

    with pytest.raises(ValidationError):
        E2BEnvironmentConfig(template_id="")
    with pytest.raises(ValidationError):
        E2BEnvironmentConfig(template_id="base-python", sandbox_timeout_s=-1)
    with pytest.raises(ValidationError):
        E2BEnvironmentConfig(template_id="base-python", request_timeout_s=-1)
    with pytest.raises(ValidationError):
        E2BEnvironmentConfig(template_id="base-python", network_policy="open")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        E2BEnvironmentConfig(template_id="base-python", startup_env={"A": 1})  # type: ignore[dict-item]
    with pytest.raises(ValidationError):
        E2BEnvironmentConfig(template_id="base-python", unexpected=True)  # type: ignore[call-arg]


@pytest.mark.fast
def test_e2b_provider_preflight_maps_schema_errors_without_leaking_values() -> None:
    provider = E2BEnvironmentProvider(client=FakeE2BClient())

    with pytest.raises(EnvironmentPreflightError) as excinfo:
        asyncio.run(
            provider.preflight(
                kit_id="tau2",
                provider="e2b",
                profile_id="e2b-profile",
                profile=_profile(),
                provider_config={"template_id": "", "startup_env": {"/secret/key": 1}},
                resources=_resources(),
                startup_env={},
                lifecycle="per_sample",
                metadata={},
            )
        )

    message = str(excinfo.value)
    assert "e2b.config validation failed" in message
    assert "template_id" in message
    assert "startup_env" in message
    assert "/secret/key" not in message
    assert "input_value" not in message


@pytest.mark.fast
def test_e2b_provider_create_passes_template_env_metadata_timeout_and_network_policy() -> None:
    fake_client = FakeE2BClient()
    provider = E2BEnvironmentProvider(client=fake_client)

    environment = asyncio.run(
        _create_environment(
            provider,
            E2BEnvironmentConfig(
                template_id="base-python",
                startup_env={"CONFIG_LEVEL": "config"},
                sandbox_timeout_s=45,
                user="agent",
            ),
            resources=_resources(network_policy="allow"),
            startup_env={"RUN_LEVEL": "request"},
        )
    )
    description = asyncio.run(environment.describe())

    assert isinstance(environment, BaseEnvironment)
    assert environment.env_id.startswith("e2b-")
    assert environment.provider == "e2b"
    assert environment.capabilities == EnvironmentCapabilities(
        supports_mounts=False,
        supports_upload_download=True,
        supports_internet_control=True,
        supports_privileged_dind=False,
        default_user="agent",
    )
    assert fake_client.create_calls == [
        {
            "template": "base-python",
            "timeout": 45,
            "metadata": {"profile_id": "e2b-profile", "sample_id": "sample-1"},
            "envs": {"CONFIG_LEVEL": "config", "RUN_LEVEL": "request"},
            "allow_internet_access": True,
        }
    ]
    assert description["diagnostics"]["sandbox_id"] == "sbx_123"
    assert description["diagnostics"]["template_id"] == "base-python"
    assert description["diagnostics"]["network_policy"] == "allow"
    assert description["diagnostics"]["allow_internet_access"] is True
    assert description["diagnostics"]["persistence_supported"] is False
    assert description["diagnostics"]["last_stop_mode"] is None
    assert description["diagnostics"]["last_attach_mode"] == "fresh"


@pytest.mark.fast
def test_e2b_config_network_policy_overrides_resource_policy_and_blocks_internet() -> None:
    fake_client = FakeE2BClient()
    provider = E2BEnvironmentProvider(client=fake_client)

    asyncio.run(
        _create_environment(
            provider,
            E2BEnvironmentConfig(template_id="base-python", network_policy="block"),
            resources=_resources(network_policy="allow"),
        )
    )

    assert fake_client.create_calls[0]["allow_internet_access"] is False


@pytest.mark.fast
def test_e2b_exec_adapts_object_results_and_passes_e2b_run_options() -> None:
    fake_sandbox = FakeSandbox()
    fake_sandbox.commands.results.append(
        ObjectCommandResult(exit_code=2, stdout=b"abcdef", stderr="problem", error="failed")
    )
    provider = E2BEnvironmentProvider(client=FakeE2BClient(fake_sandbox))
    environment = asyncio.run(
        _create_environment(
            provider,
            E2BEnvironmentConfig(
                template_id="base-python",
                request_timeout_s=7,
                user="agent",
                stdout_limit_bytes=4,
            ),
        )
    )

    result = asyncio.run(
        environment.exec("printf abcdef", cwd="/workspace", env={"A": "1"}, timeout_s=3, user="runner")
    )

    assert result.command == "printf abcdef"
    assert result.exit_code == 2
    assert result.stdout == "abcd"
    assert result.stderr == "problem"
    assert result.truncated is True
    assert result.metadata["error"] == "failed"
    assert fake_sandbox.commands.run_calls == [
        {
            "cmd": "printf abcdef",
            "envs": {"A": "1"},
            "cwd": "/workspace",
            "timeout": 3,
            "request_timeout": 7,
            "user": "runner",
        }
    ]


@pytest.mark.fast
def test_e2b_exec_adapts_command_exit_exception_as_nonzero_result() -> None:
    fake_sandbox = FakeSandbox()
    fake_sandbox.commands.results.append(
        CommandExitException(exit_code=1, stdout="before failure\n", stderr=b"failed\n", error="exit status 1")
    )
    provider = E2BEnvironmentProvider(client=FakeE2BClient(fake_sandbox))
    environment = asyncio.run(_create_environment(provider, E2BEnvironmentConfig(template_id="base-python")))

    result = asyncio.run(environment.exec("false"))

    assert result.command == "false"
    assert result.exit_code == 1
    assert result.stdout == "before failure\n"
    assert result.stderr == "failed\n"
    assert result.metadata["error"] == "exit status 1"


@pytest.mark.fast
def test_e2b_command_exit_exception_with_timed_out_stderr_is_not_timeout() -> None:
    fake_sandbox = FakeSandbox()
    fake_sandbox.commands.results.append(
        CommandExitException(
            exit_code=1,
            stderr="timed out but normal exit",
            error="exit status 1",
        )
    )
    provider = E2BEnvironmentProvider(client=FakeE2BClient(fake_sandbox))
    environment = asyncio.run(_create_environment(provider, E2BEnvironmentConfig(template_id="base-python")))

    result = asyncio.run(environment.exec("printf timed-out-message; exit 1"))

    assert result.exit_code == 1
    assert result.stderr == "timed out but normal exit"
    assert result.metadata["error"] == "exit status 1"


@pytest.mark.fast
def test_e2b_file_probes_return_false_when_command_exit_exception_has_exit_code_one() -> None:
    fake_sandbox = FakeSandbox()
    fake_sandbox.commands.results.extend(
        [
            CommandExitException(exit_code=1, stderr="missing"),
            CommandExitException(exit_code=1, stderr="missing"),
        ]
    )
    provider = E2BEnvironmentProvider(client=FakeE2BClient(fake_sandbox))
    environment = asyncio.run(_create_environment(provider, E2BEnvironmentConfig(template_id="base-python")))

    assert asyncio.run(environment.is_file("/missing")) is False
    assert asyncio.run(environment.is_dir("/missing")) is False


@pytest.mark.fast
def test_e2b_exec_maps_timed_out_result_and_timeout_exceptions() -> None:
    fake_sandbox = FakeSandbox()
    fake_sandbox.commands.results.extend(
        [
            {"exit_code": None, "stdout": "", "stderr": "late", "timed_out": True},
            TimeoutError("request timed out"),
        ]
    )
    provider = E2BEnvironmentProvider(client=FakeE2BClient(fake_sandbox))
    environment = asyncio.run(_create_environment(provider, E2BEnvironmentConfig(template_id="base-python")))

    with pytest.raises(EnvironmentTimeoutError, match="e2b.exec timeout"):
        asyncio.run(environment.exec("sleep 10", timeout_s=1))
    with pytest.raises(EnvironmentTimeoutError, match="e2b.exec timeout"):
        asyncio.run(environment.exec("sleep 10", timeout_s=1))


@pytest.mark.fast
def test_e2b_file_methods_cover_read_write_upload_download_and_limits(tmp_path: Path) -> None:
    fake_sandbox = FakeSandbox()
    provider = E2BEnvironmentProvider(client=FakeE2BClient(fake_sandbox))
    environment = asyncio.run(
        _create_environment(
            provider,
            E2BEnvironmentConfig(template_id="base-python", request_timeout_s=9, user="agent"),
        )
    )

    asyncio.run(environment.write_file("/workspace/written.txt", "written"))
    assert fake_sandbox.files.storage["/workspace/written.txt"] == b"written"
    assert fake_sandbox.files.write_calls[0]["request_timeout"] == 9
    assert fake_sandbox.files.write_calls[0]["user"] == "agent"
    assert asyncio.run(environment.read_file("/workspace/written.txt")) == b"written"
    assert fake_sandbox.files.read_calls[0]["format"] == "bytes"

    upload = tmp_path / "upload.txt"
    upload.write_text("uploaded", encoding="utf-8")
    asyncio.run(environment.upload_file(upload, "/workspace/upload.txt"))
    assert fake_sandbox.files.storage["/workspace/upload.txt"] == b"uploaded"

    download = tmp_path / "download.txt"
    asyncio.run(environment.download_file("/workspace/upload.txt", download))
    assert download.read_text(encoding="utf-8") == "uploaded"

    with pytest.raises(EnvironmentTransferError, match="read_file.size_limit"):
        asyncio.run(environment.read_file("/workspace/upload.txt", max_bytes=3))
    with pytest.raises(EnvironmentFileNotFoundError):
        asyncio.run(environment.read_file("/workspace/missing.txt"))


@pytest.mark.fast
def test_e2b_download_file_is_not_limited_by_read_file_max_bytes(tmp_path: Path) -> None:
    fake_sandbox = FakeSandbox()
    payload = b"x" * (DEFAULT_READ_FILE_LIMIT_BYTES + 1)
    fake_sandbox.files.storage["/workspace/large.txt"] = payload
    provider = E2BEnvironmentProvider(client=FakeE2BClient(fake_sandbox))
    environment = asyncio.run(_create_environment(provider, E2BEnvironmentConfig(template_id="base-python")))

    with pytest.raises(EnvironmentTransferError, match="read_file.size_limit"):
        asyncio.run(environment.read_file("/workspace/large.txt", max_bytes=3))

    target = tmp_path / "large.txt"
    asyncio.run(environment.download_file("/workspace/large.txt", target))

    assert target.read_bytes() == payload


@pytest.mark.fast
def test_e2b_download_dir_does_not_replace_local_target_when_remote_is_missing(tmp_path: Path) -> None:
    fake_sandbox = FakeSandbox()
    fake_sandbox.commands.results.append(ExecResult(command="test-dir", exit_code=1))
    provider = E2BEnvironmentProvider(client=FakeE2BClient(fake_sandbox))
    environment = asyncio.run(_create_environment(provider, E2BEnvironmentConfig(template_id="base-python")))
    target = tmp_path / "downloaded"
    target.mkdir()
    existing = target / "keep.txt"
    existing.write_text("keep", encoding="utf-8")

    with pytest.raises(EnvironmentFileNotFoundError):
        asyncio.run(environment.download_dir("/workspace/missing", target))

    assert existing.read_text(encoding="utf-8") == "keep"
    assert target.is_dir()


@pytest.mark.fast
def test_e2b_download_dir_does_not_replace_local_file_when_remote_is_regular_file(tmp_path: Path) -> None:
    fake_sandbox = FakeSandbox()
    fake_sandbox.commands.results.append(ExecResult(command="test-dir", exit_code=1))
    provider = E2BEnvironmentProvider(client=FakeE2BClient(fake_sandbox))
    environment = asyncio.run(_create_environment(provider, E2BEnvironmentConfig(template_id="base-python")))
    target = tmp_path / "downloaded"
    target.write_text("local", encoding="utf-8")

    with pytest.raises(EnvironmentFileNotFoundError):
        asyncio.run(environment.download_dir("/workspace/file.txt", target))

    assert target.read_text(encoding="utf-8") == "local"


@pytest.mark.fast
def test_e2b_list_files_rejects_regular_file_and_missing_path() -> None:
    fake_sandbox = FakeSandbox()
    fake_sandbox.commands = PathAwareCommands()
    provider = E2BEnvironmentProvider(client=FakeE2BClient(fake_sandbox))
    environment = asyncio.run(_create_environment(provider, E2BEnvironmentConfig(template_id="base-python")))

    with pytest.raises(EnvironmentFileNotFoundError):
        asyncio.run(environment.list_files("/workspace/file.txt"))
    with pytest.raises(EnvironmentFileNotFoundError):
        asyncio.run(environment.list_files("/workspace/missing"))

    commands = [call["cmd"] for call in fake_sandbox.commands.run_calls]
    assert commands == ["test -d /workspace/file.txt", "test -d /workspace/missing"]


@pytest.mark.fast
def test_e2b_list_and_file_probes_quote_remote_paths() -> None:
    fake_sandbox = FakeSandbox()
    fake_sandbox.commands.results.extend(
        [
            ExecResult(command="test-dir", exit_code=0),
            ExecResult(command="find", exit_code=0, stdout="f 12 /workspace/has spaces; touch /pwned/file.txt\n"),
            ExecResult(command="test-file", exit_code=0),
            ExecResult(command="test-dir", exit_code=1),
        ]
    )
    provider = E2BEnvironmentProvider(client=FakeE2BClient(fake_sandbox))
    environment = asyncio.run(_create_environment(provider, E2BEnvironmentConfig(template_id="base-python")))
    remote_path = "/workspace/has spaces; touch /pwned"
    quoted_path = shlex.quote(remote_path)

    entries = asyncio.run(environment.list_files(remote_path))
    assert asyncio.run(environment.is_file(remote_path)) is True
    assert asyncio.run(environment.is_dir(remote_path)) is False

    assert [(entry.path, entry.kind, entry.size_bytes) for entry in entries] == [
        ("/workspace/has spaces; touch /pwned/file.txt", "file", 12)
    ]
    commands = [call["cmd"] for call in fake_sandbox.commands.run_calls]
    assert commands[0] == f"test -d {quoted_path}"
    assert commands[1].startswith(f"find {quoted_path} ")
    assert commands[2] == f"test -f {quoted_path}"
    assert commands[3] == f"test -d {quoted_path}"
    assert f"test -d {remote_path}" not in commands[0]
    assert f"find {remote_path} " not in commands[1]
    assert f"test -f {remote_path}" not in commands[2]
    assert f"test -d {remote_path}" not in commands[3]


@pytest.mark.fast
def test_e2b_stop_kills_sandbox_and_updates_persistence_diagnostics() -> None:
    fake_sandbox = FakeSandbox()
    provider = E2BEnvironmentProvider(client=FakeE2BClient(fake_sandbox))
    environment = asyncio.run(_create_environment(provider, E2BEnvironmentConfig(template_id="base-python")))

    asyncio.run(environment.stop(delete=True))
    description = asyncio.run(environment.describe())

    assert fake_sandbox.kill_calls == 1
    assert description["diagnostics"]["persistence_supported"] is False
    assert description["diagnostics"]["last_stop_mode"] == "deleted"


@pytest.mark.fast
def test_opensandbox_provider_registered_as_reserved_stub() -> None:
    registry = create_default_provider_registry()
    provider = registry.get("opensandbox")
    request = {
        "kit_id": "tau2",
        "provider": "opensandbox",
        "profile_id": "opensandbox-profile",
        "profile": EnvironmentProfile(profile_id="opensandbox-profile", provider="opensandbox", config={}),
        "provider_config": {},
        "resources": _resources(),
        "startup_env": {},
        "lifecycle": "per_sample",
        "metadata": {},
    }

    assert "opensandbox" in registry.registered_provider_ids()
    with pytest.raises(EnvironmentCreateError) as create_excinfo:
        asyncio.run(provider.create(**request))
    with pytest.raises(EnvironmentCreateError) as acquire_excinfo:
        asyncio.run(provider.acquire(**request))

    for message in (str(create_excinfo.value).lower(), str(acquire_excinfo.value).lower()):
        assert "opensandbox.provider.unsupported" in message
        assert "daytona" not in message
        assert "e2b" not in message


@pytest.mark.fast
def test_opensandbox_reserved_provider_manager_acquire_does_not_retry() -> None:
    registry = create_default_provider_registry()
    backoffs: list[dict[str, Any]] = []

    async def record_backoff(*, provider: str, attempt: int, delay_s: float, error: BaseException) -> None:
        del error
        backoffs.append({"provider": provider, "attempt": attempt, "delay_s": delay_s})

    manager = EnvironmentManager(registry=registry, backoff=record_backoff)

    with pytest.raises(EnvironmentManagerError) as excinfo:
        asyncio.run(
            manager.acquire(
                kit_id="tau2",
                provider="opensandbox",
                profile_id="opensandbox-profile",
                profile=EnvironmentProfile(profile_id="opensandbox-profile", provider="opensandbox", config={}),
                provider_config={},
                resources=_resources(),
                startup_env={},
                lifecycle="per_sample",
                metadata={"sample_id": "sample-1"},
            )
        )

    assert "environment.create_failed" in str(excinfo.value)
    assert "attempts=1" in str(excinfo.value)
    assert "opensandbox.provider.unsupported" in str(excinfo.value.__cause__)
    assert backoffs == []


@pytest.mark.fast
def test_e2b_sdk_imports_are_confined_to_e2b_provider_directory() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source_root = repo_root / "src" / "gage_eval"
    offenders: list[str] = []
    pattern = re.compile(r"^\s*(?:from\s+e2b\b|import\s+e2b\b)", re.MULTILINE)

    for path in source_root.rglob("*.py"):
        if not pattern.search(path.read_text(encoding="utf-8")):
            continue
        if "environment/providers/e2b" not in path.as_posix():
            offenders.append(str(path.relative_to(repo_root)))

    assert offenders == []
