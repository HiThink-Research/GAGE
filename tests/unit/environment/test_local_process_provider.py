from __future__ import annotations

import asyncio
import os
import shlex
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from gage_eval.environment import (
    BaseEnvironment,
    EnvironmentCapabilities,
    EnvironmentFileNotFoundError,
    EnvironmentPreflightError,
    EnvironmentResources,
    EnvironmentTimeoutError,
    EnvironmentTransferError,
)
from gage_eval.environment.profiles import EnvironmentProfile
from gage_eval.environment.providers.local_process import (
    LocalProcessEnvironmentConfig,
    LocalProcessEnvironmentProvider,
)


def _profile(config: dict[str, Any] | None = None) -> EnvironmentProfile:
    return EnvironmentProfile(profile_id="local-profile", provider="local_process", config=config or {})


def _resources() -> EnvironmentResources:
    return EnvironmentResources(cpu=1.0, memory_gb=2.0, network_policy="block")


async def _create_environment(
    provider: LocalProcessEnvironmentProvider,
    config: LocalProcessEnvironmentConfig | dict[str, Any],
    *,
    profile: EnvironmentProfile | None = None,
    startup_env: dict[str, str] | None = None,
) -> BaseEnvironment:
    return await provider.create(
        kit_id="tau2",
        provider="local_process",
        profile_id="local-profile",
        profile=profile or _profile(),
        provider_config=config,
        resources=_resources(),
        startup_env=startup_env or {},
        lifecycle="per_sample",
        metadata={"sample_id": "sample-1"},
    )


@pytest.mark.fast
def test_local_process_config_schema_validates_paths_env_and_stream_limits(tmp_path: Path) -> None:
    config = LocalProcessEnvironmentConfig(
        base_cwd=str(tmp_path),
        startup_env={"CONFIG_LEVEL": "1"},
        stdout_limit_bytes=16,
        stderr_limit_bytes=8,
    )

    assert config.base_cwd == str(tmp_path)
    assert config.workdir is None
    assert config.startup_env == {"CONFIG_LEVEL": "1"}
    assert config.stdout_limit_bytes == 16
    assert config.stderr_limit_bytes == 8

    with pytest.raises(ValidationError):
        LocalProcessEnvironmentConfig(base_cwd="relative")
    with pytest.raises(ValidationError):
        LocalProcessEnvironmentConfig(workdir="")
    with pytest.raises(ValidationError):
        LocalProcessEnvironmentConfig(workdir=str(tmp_path / "work"), base_cwd=str(tmp_path))
    with pytest.raises(ValidationError):
        LocalProcessEnvironmentConfig(stdout_limit_bytes=-1)
    with pytest.raises(ValidationError):
        LocalProcessEnvironmentConfig(startup_env={"CONFIG_LEVEL": 1})  # type: ignore[dict-item]
    with pytest.raises(ValidationError):
        LocalProcessEnvironmentConfig(base_cwd=str(tmp_path), unexpected=True)  # type: ignore[call-arg]


@pytest.mark.fast
def test_local_process_provider_preflight_maps_schema_errors_to_environment_error(tmp_path: Path) -> None:
    provider = LocalProcessEnvironmentProvider()

    with pytest.raises(EnvironmentPreflightError, match="local_process.config"):
        asyncio.run(
            provider.preflight(
                kit_id="tau2",
                provider="local_process",
                profile_id="local-profile",
                profile=_profile({"base_cwd": str(tmp_path)}),
                provider_config={"workdir": "relative"},
                resources=_resources(),
                startup_env={},
                lifecycle="per_sample",
                metadata={},
            )
        )


@pytest.mark.fast
def test_local_process_provider_preflight_validation_error_does_not_leak_host_paths(tmp_path: Path) -> None:
    provider = LocalProcessEnvironmentProvider()
    workdir = tmp_path / "work"
    base_cwd = tmp_path / "base"

    with pytest.raises(EnvironmentPreflightError) as excinfo:
        asyncio.run(
            provider.preflight(
                kit_id="tau2",
                provider="local_process",
                profile_id="local-profile",
                profile=_profile(),
                provider_config={
                    "workdir": str(workdir),
                    "base_cwd": str(base_cwd),
                },
                resources=_resources(),
                startup_env={},
                lifecycle="per_sample",
                metadata={},
            )
        )

    message = str(excinfo.value)
    assert "local_process.config validation failed" in message
    assert "workdir" in message
    assert "base_cwd" in message
    assert str(tmp_path) not in message
    assert str(workdir) not in message
    assert str(base_cwd) not in message
    assert "input_value" not in message
    assert "input_type" not in message
    assert excinfo.value.__cause__ is None


@pytest.mark.fast
def test_local_process_provider_preflight_extra_key_loc_does_not_leak_host_path(tmp_path: Path) -> None:
    provider = LocalProcessEnvironmentProvider()
    secret_key = str(tmp_path / "secret-extra")

    with pytest.raises(EnvironmentPreflightError) as excinfo:
        asyncio.run(
            provider.preflight(
                kit_id="tau2",
                provider="local_process",
                profile_id="local-profile",
                profile=_profile(),
                provider_config={secret_key: "value"},
                resources=_resources(),
                startup_env={},
                lifecycle="per_sample",
                metadata={},
            )
        )

    message = str(excinfo.value)
    assert "local_process.config validation failed" in message
    assert "extra_forbidden" in message
    assert "<redacted>" in message
    assert str(tmp_path) not in message
    assert "secret-extra" not in message
    assert excinfo.value.__cause__ is None


@pytest.mark.fast
def test_local_process_provider_preflight_startup_env_key_loc_does_not_leak_host_path(
    tmp_path: Path,
) -> None:
    provider = LocalProcessEnvironmentProvider()
    secret_key = str(tmp_path / "secret-env-key")

    with pytest.raises(EnvironmentPreflightError) as excinfo:
        asyncio.run(
            provider.preflight(
                kit_id="tau2",
                provider="local_process",
                profile_id="local-profile",
                profile=_profile(),
                provider_config={"startup_env": {secret_key: 1}},
                resources=_resources(),
                startup_env={},
                lifecycle="per_sample",
                metadata={},
            )
        )

    message = str(excinfo.value)
    assert "local_process.config validation failed" in message
    assert "startup_env" in message
    assert "<redacted>" in message
    assert str(tmp_path) not in message
    assert "secret-env-key" not in message


@pytest.mark.fast
def test_local_process_provider_create_returns_base_environment_and_describes_weak_isolation(
    tmp_path: Path,
) -> None:
    provider = LocalProcessEnvironmentProvider()
    environment = asyncio.run(_create_environment(provider, LocalProcessEnvironmentConfig(base_cwd=str(tmp_path))))
    description = asyncio.run(environment.describe())

    assert isinstance(environment, BaseEnvironment)
    assert environment.env_id.startswith("local-process-")
    assert environment.provider == "local_process"
    assert environment.capabilities == EnvironmentCapabilities(
        supports_mounts=False,
        supports_upload_download=True,
        supports_internet_control=False,
        supports_privileged_dind=False,
        default_user=None,
    )
    assert description["env_id"] == environment.env_id
    assert description["provider"] == "local_process"
    assert description["diagnostics"]["strong_isolation"] is False
    assert "does not provide strong security isolation" in description["diagnostics"]["warning"]
    assert str(tmp_path) not in repr(description)


@pytest.mark.fast
def test_local_process_exec_runs_in_workdir_and_passes_cwd_env_and_startup_env(tmp_path: Path) -> None:
    provider = LocalProcessEnvironmentProvider()
    environment = asyncio.run(
        _create_environment(
            provider,
            LocalProcessEnvironmentConfig(base_cwd=str(tmp_path), startup_env={"CONFIG_LEVEL": "config"}),
            startup_env={"REQUEST_LEVEL": "request"},
        )
    )
    asyncio.run(environment.write_file("/run/.keep", ""))

    script = (
        "import os, pathlib; "
        "pathlib.Path('env.txt').write_text("
        "os.environ['CONFIG_LEVEL'] + ':' + os.environ['REQUEST_LEVEL'] + ':' + os.environ['CALL_LEVEL'], "
        "encoding='utf-8')"
    )
    result = asyncio.run(
        environment.exec(
            f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}",
            cwd="/run",
            env={"CALL_LEVEL": "call"},
            timeout_s=5,
        )
    )

    assert result.exit_code == 0
    assert result.timed_out is False
    assert asyncio.run(environment.read_file("/run/env.txt")) == b"config:request:call"


@pytest.mark.fast
def test_local_process_exec_timeout_raises_environment_timeout_error(tmp_path: Path) -> None:
    provider = LocalProcessEnvironmentProvider()
    environment = asyncio.run(_create_environment(provider, LocalProcessEnvironmentConfig(base_cwd=str(tmp_path))))
    script = "import time; time.sleep(5)"

    with pytest.raises(EnvironmentTimeoutError, match="local_process.exec timeout"):
        asyncio.run(environment.exec(f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}", timeout_s=0))


@pytest.mark.fast
def test_local_process_timeout_cleanup_does_not_wait_for_detached_child_holding_pipe(
    tmp_path: Path,
) -> None:
    provider = LocalProcessEnvironmentProvider()
    environment = asyncio.run(_create_environment(provider, LocalProcessEnvironmentConfig(base_cwd=str(tmp_path))))
    child_pid: int | None = None
    child_code = (
        "import os, pathlib, time; "
        "pathlib.Path('detached-child.pid').write_text(str(os.getpid()), encoding='utf-8'); "
        "time.sleep(4)"
    )
    parent_code = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], start_new_session=True); "
        "time.sleep(30)"
    )

    started_at = time.perf_counter()
    try:
        with pytest.raises(EnvironmentTimeoutError, match="local_process.exec timeout"):
            asyncio.run(environment.exec(f"{shlex.quote(sys.executable)} -c {shlex.quote(parent_code)}", timeout_s=1))
        elapsed_s = time.perf_counter() - started_at
        assert elapsed_s < 2.5
    finally:
        try:
            child_pid = int(asyncio.run(environment.read_file("/detached-child.pid")).decode("utf-8"))
        except Exception:
            child_pid = None
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.fast
def test_local_process_shell_none_splits_without_shell_expansion(tmp_path: Path) -> None:
    provider = LocalProcessEnvironmentProvider()
    environment = asyncio.run(_create_environment(provider, LocalProcessEnvironmentConfig(base_cwd=str(tmp_path))))
    script = "import sys; print(sys.argv[1])"

    result = asyncio.run(
        environment.exec(
            f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} '$HOME'",
            shell="none",
        )
    )

    assert result.exit_code == 0
    assert result.stdout == "$HOME\n"


@pytest.mark.fast
def test_local_process_file_methods_cover_read_write_list_is_file_is_dir_and_transfers(
    tmp_path: Path,
) -> None:
    provider = LocalProcessEnvironmentProvider()
    environment = asyncio.run(_create_environment(provider, LocalProcessEnvironmentConfig(base_cwd=str(tmp_path))))

    asyncio.run(environment.write_file("/nested/file.txt", "payload"))
    assert asyncio.run(environment.read_file("/nested/file.txt")) == b"payload"
    assert asyncio.run(environment.is_file("/nested/file.txt")) is True
    assert asyncio.run(environment.is_dir("/nested")) is True
    assert asyncio.run(environment.is_file("/nested")) is False
    assert asyncio.run(environment.is_dir("/nested/file.txt")) is False

    entries = asyncio.run(environment.list_files("/nested"))
    assert [(entry.path, entry.kind, entry.size_bytes) for entry in entries] == [
        ("/nested/file.txt", "file", len(b"payload"))
    ]

    local_upload = tmp_path / "upload.txt"
    local_upload.write_text("uploaded", encoding="utf-8")
    asyncio.run(environment.upload_file(local_upload, "/nested/upload.txt"))
    assert asyncio.run(environment.read_file("/nested/upload.txt")) == b"uploaded"

    local_dir = tmp_path / "local-dir"
    local_dir.mkdir()
    (local_dir / "child.txt").write_text("dir payload", encoding="utf-8")
    asyncio.run(environment.upload_dir(local_dir, "/uploaded-dir"))
    assert asyncio.run(environment.read_file("/uploaded-dir/child.txt")) == b"dir payload"

    downloaded_file = tmp_path / "download.txt"
    asyncio.run(environment.download_file("/nested/upload.txt", downloaded_file))
    assert downloaded_file.read_text(encoding="utf-8") == "uploaded"

    downloaded_dir = tmp_path / "downloaded-dir"
    asyncio.run(environment.download_dir("/uploaded-dir", downloaded_dir))
    assert (downloaded_dir / "child.txt").read_text(encoding="utf-8") == "dir payload"

    with pytest.raises(EnvironmentFileNotFoundError):
        asyncio.run(environment.read_file("/missing.txt"))


@pytest.mark.fast
def test_local_process_root_path_maps_to_provider_workdir(tmp_path: Path) -> None:
    provider = LocalProcessEnvironmentProvider()
    environment = asyncio.run(_create_environment(provider, LocalProcessEnvironmentConfig(base_cwd=str(tmp_path))))

    asyncio.run(environment.write_file("/root.txt", "root"))

    entries = asyncio.run(environment.list_files("/"))
    assert ("/root.txt", "file", len(b"root")) in [
        (entry.path, entry.kind, entry.size_bytes) for entry in entries
    ]
    assert asyncio.run(environment.is_dir("/")) is True
    assert asyncio.run(environment.is_file("/")) is False
    assert asyncio.run(environment.read_file("/root.txt")) == b"root"

    script = "import os, pathlib; pathlib.Path('cwd.txt').write_text(os.getcwd(), encoding='utf-8')"
    result = asyncio.run(environment.exec(f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}", cwd="/"))
    assert result.exit_code == 0

    cwd = Path(asyncio.run(environment.read_file("/cwd.txt")).decode("utf-8"))
    assert cwd.parent.resolve() == tmp_path.resolve()


@pytest.mark.fast
def test_local_process_read_limit_and_unsafe_paths_are_environment_errors(tmp_path: Path) -> None:
    provider = LocalProcessEnvironmentProvider()
    environment = asyncio.run(_create_environment(provider, LocalProcessEnvironmentConfig(base_cwd=str(tmp_path))))
    asyncio.run(environment.write_file("/large.txt", b"abcdef"))

    with pytest.raises(EnvironmentTransferError, match="read_file.size_limit"):
        asyncio.run(environment.read_file("/large.txt", max_bytes=3))
    with pytest.raises(EnvironmentTransferError) as read_excinfo:
        asyncio.run(environment.read_file("../outside.txt"))
    with pytest.raises(EnvironmentTransferError) as write_excinfo:
        asyncio.run(environment.write_file("/../outside.txt", "outside"))
    with pytest.raises(EnvironmentTransferError) as list_excinfo:
        asyncio.run(environment.list_files("../outside"))

    for excinfo in (read_excinfo, write_excinfo, list_excinfo):
        message = str(excinfo.value)
        assert ".." in message
        assert str(tmp_path) not in message


@pytest.mark.fast
def test_local_process_rejects_file_api_symlink_escape(tmp_path: Path) -> None:
    provider = LocalProcessEnvironmentProvider()
    environment = asyncio.run(_create_environment(provider, LocalProcessEnvironmentConfig(base_cwd=str(tmp_path))))
    asyncio.run(environment.write_file("/inside.txt", "inside"))

    description = asyncio.run(environment.describe())
    assert str(tmp_path) not in repr(description)

    script = (
        "import os, pathlib; "
        "pathlib.Path('..', 'outside.txt').write_text('outside', encoding='utf-8'); "
        "os.symlink('../outside.txt', 'link.txt')"
    )
    assert asyncio.run(environment.exec(f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}")).exit_code == 0

    with pytest.raises(EnvironmentTransferError, match="symlink"):
        asyncio.run(environment.read_file("/link.txt"))
    with pytest.raises(EnvironmentTransferError, match="symlink"):
        asyncio.run(environment.write_file("/link.txt", "modified"))

    outside_path = tmp_path / "outside.txt"
    assert outside_path.read_text(encoding="utf-8") == "outside"


@pytest.mark.fast
def test_local_process_large_stdout_is_truncated_with_configured_limit(tmp_path: Path) -> None:
    provider = LocalProcessEnvironmentProvider()
    environment = asyncio.run(
        _create_environment(
            provider,
            LocalProcessEnvironmentConfig(base_cwd=str(tmp_path), stdout_limit_bytes=5, stderr_limit_bytes=5),
        )
    )
    script = "import sys; sys.stdout.write('abcdefghijklmnopqrstuvwxyz')"

    result = asyncio.run(environment.exec(f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"))

    assert result.exit_code == 0
    assert result.stdout == "abcde"
    assert result.stderr == ""
    assert result.truncated is True
    assert result.output_artifact_refs == []
