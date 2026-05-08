from __future__ import annotations

import asyncio
import io
import shlex
import tarfile
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from gage_eval.environment import (
    BaseEnvironment,
    EnvironmentCapabilities,
    EnvironmentCreateError,
    EnvironmentFileNotFoundError,
    EnvironmentPreflightError,
    EnvironmentResources,
    EnvironmentTransferError,
    ExecResult,
)
from gage_eval.environment.profiles import EnvironmentProfile
from gage_eval.environment.providers.docker import (
    DockerEnvironmentConfig,
    DockerEnvironmentProvider,
    DockerMount,
)
from gage_eval.environment.providers.docker.provider import _DockerSDKContainerAdapter


class FakeDockerClient:
    def __init__(self, container: "FakeContainer | None" = None) -> None:
        self.container = container or FakeContainer()
        self.create_calls: list[dict[str, Any]] = []
        self.build_calls: list[dict[str, Any]] = []
        self.ping_calls = 0

    def ping(self) -> None:
        self.ping_calls += 1

    def create_container(self, **kwargs: Any) -> "FakeContainer":
        self.create_calls.append(kwargs)
        return self.container

    def build_image(self, **kwargs: Any) -> None:
        self.build_calls.append(kwargs)


class FakeContainer:
    id = "container-123"
    name = "gage-env-container-123"

    def __init__(self) -> None:
        self.exec_calls: list[dict[str, Any]] = []
        self.put_archive_calls: list[dict[str, Any]] = []
        self.get_archive_calls: list[str] = []
        self.files: dict[str, bytes] = {}
        self.stop_calls: list[bool] = []

    def exec(self, **kwargs: Any) -> ExecResult:
        self.exec_calls.append(kwargs)
        command = kwargs["command"]
        rendered = command if isinstance(command, str) else " ".join(command)
        return ExecResult(command=rendered, exit_code=0, stdout="ok\n", stderr="", duration_s=0.01)

    def put_archive(self, remote_dir: str, data: bytes) -> None:
        self.put_archive_calls.append({"remote_dir": remote_dir, "data": data})
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            for member in archive.getmembers():
                if member.isdir():
                    continue
                extracted = archive.extractfile(member)
                assert extracted is not None
                self.files[_remote_join(remote_dir, member.name)] = extracted.read()

    def get_archive(self, remote_path: str) -> bytes:
        self.get_archive_calls.append(remote_path)
        if remote_path not in self.files:
            raise FileNotFoundError(remote_path)
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w") as archive:
            content = self.files[remote_path]
            info = tarfile.TarInfo(name=Path(remote_path).name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        return payload.getvalue()

    def stop(self, *, delete: bool = True) -> None:
        self.stop_calls.append(delete)

    def logs(self, *, stream: str | None = None) -> str:
        return f"logs:{stream or 'all'}"


def _remote_join(parent: str, child: str) -> str:
    return f"{parent.rstrip('/')}/{child.lstrip('/')}"


def _profile(config: dict[str, Any] | None = None) -> EnvironmentProfile:
    return EnvironmentProfile(profile_id="docker-profile", provider="docker", config=config or {})


def _resources(network_policy: str = "block") -> EnvironmentResources:
    return EnvironmentResources(cpu=1.0, memory_gb=2.0, network_policy=network_policy)  # type: ignore[arg-type]


async def _create_environment(
    provider: DockerEnvironmentProvider,
    config: DockerEnvironmentConfig | dict[str, Any],
    *,
    profile: EnvironmentProfile | None = None,
    resources: EnvironmentResources | None = None,
) -> BaseEnvironment:
    return await provider.create(
        kit_id="tau2",
        provider="docker",
        profile_id="docker-profile",
        profile=profile or _profile(),
        provider_config=config,
        resources=resources or _resources(),
        startup_env={"RUN_LEVEL": "1"},
        lifecycle="per_sample",
        metadata={"sample_id": "sample-1"},
    )


@pytest.mark.fast
def test_docker_config_schema_validates_image_privileged_platform_mounts_and_network_policy() -> None:
    config = DockerEnvironmentConfig(
        image="python:3.12-slim",
        privileged=True,
        docker_platform="linux/arm64",
        network_policy="egress_only",
        mounts=[DockerMount(source="/host/cache", target="/cache", read_only=True)],
    )

    assert config.image == "python:3.12-slim"
    assert config.privileged is True
    assert config.docker_platform == "linux/arm64"
    assert config.network_policy == "egress_only"
    assert config.mounts[0].target == "/cache"

    with pytest.raises(ValidationError):
        DockerEnvironmentConfig(image="")
    with pytest.raises(ValidationError):
        DockerEnvironmentConfig(image="python:3.12-slim", privileged="true")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        DockerEnvironmentConfig(image="python:3.12-slim", docker_platform="")
    with pytest.raises(ValidationError):
        DockerEnvironmentConfig(image="python:3.12-slim", network_policy="open")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        DockerEnvironmentConfig(
            image="python:3.12-slim",
            mounts=[DockerMount(source="/host/cache", target="relative")],
        )


@pytest.mark.fast
def test_docker_provider_preflight_maps_schema_errors_to_environment_error() -> None:
    provider = DockerEnvironmentProvider(client=FakeDockerClient())

    with pytest.raises(EnvironmentPreflightError, match="docker.config"):
        asyncio.run(
            provider.preflight(
                kit_id="tau2",
                provider="docker",
                profile_id="docker-profile",
                profile=_profile(),
                provider_config={"image": ""},
                resources=_resources(),
                startup_env={},
                lifecycle="per_sample",
                metadata={},
            )
        )


@pytest.mark.fast
def test_docker_provider_create_returns_base_environment_contract() -> None:
    fake_client = FakeDockerClient()
    provider = DockerEnvironmentProvider(client=fake_client)

    environment = asyncio.run(
        _create_environment(
            provider,
            DockerEnvironmentConfig(image="python:3.12-slim", user="agent"),
            resources=_resources(network_policy="block"),
        )
    )
    description = asyncio.run(environment.describe())

    assert isinstance(environment, BaseEnvironment)
    assert environment.env_id.startswith("docker-")
    assert environment.provider == "docker"
    assert environment.capabilities == EnvironmentCapabilities(
        supports_mounts=False,
        supports_upload_download=True,
        supports_internet_control=False,
        supports_privileged_dind=False,
        default_user="agent",
    )
    assert description["env_id"] == environment.env_id
    assert description["provider"] == "docker"
    assert description["diagnostics"]["container_id"] == "container-123"
    assert "host_workdir" not in repr(description)


@pytest.mark.fast
def test_docker_provider_passes_docker_platform_to_create() -> None:
    fake_client = FakeDockerClient()
    provider = DockerEnvironmentProvider(client=fake_client)

    asyncio.run(
        _create_environment(
            provider,
            {"image": "python:3.12-slim"},
            profile=_profile({"docker_platform": "linux/arm64"}),
        )
    )

    assert fake_client.create_calls[0]["platform"] == "linux/arm64"


@pytest.mark.fast
def test_docker_provider_passes_entrypoint_keepalive_and_exec_workdir() -> None:
    fake_container = FakeContainer()
    fake_client = FakeDockerClient(fake_container)
    provider = DockerEnvironmentProvider(client=fake_client)

    environment = asyncio.run(
        _create_environment(
            provider,
            DockerEnvironmentConfig(
                image="python:3.12-slim",
                entrypoint=[],
                keepalive_command=["sleep", "infinity"],
                exec_workdir="/app",
            ),
        )
    )
    asyncio.run(environment.exec("pwd"))

    create_call = fake_client.create_calls[0]
    assert create_call["entrypoint"] == []
    assert create_call["command"] == ["sleep", "infinity"]
    assert fake_container.exec_calls[0]["cwd"] == "/app"


@pytest.mark.fast
def test_docker_provider_passes_docker_platform_to_build_when_forced() -> None:
    fake_client = FakeDockerClient()
    provider = DockerEnvironmentProvider(client=fake_client)
    environment = asyncio.run(
        _create_environment(
            provider,
            DockerEnvironmentConfig(image="python:3.12-slim", docker_platform="linux/arm64"),
        )
    )

    asyncio.run(environment.start(force_build=True))

    assert fake_client.build_calls == [{"image": "python:3.12-slim", "platform": "linux/arm64"}]


@pytest.mark.fast
def test_docker_provider_uses_resource_network_policy_when_config_omits_policy() -> None:
    fake_client = FakeDockerClient()
    provider = DockerEnvironmentProvider(client=fake_client)

    asyncio.run(
        _create_environment(
            provider,
            {"image": "python:3.12-slim"},
            resources=_resources(network_policy="allow"),
        )
    )

    assert fake_client.create_calls[0]["network_policy"] == "allow"
    assert fake_client.create_calls[0]["network_mode"] == "bridge"


@pytest.mark.fast
def test_docker_provider_explicit_config_network_policy_overrides_resources() -> None:
    fake_client = FakeDockerClient()
    provider = DockerEnvironmentProvider(client=fake_client)

    asyncio.run(
        _create_environment(
            provider,
            DockerEnvironmentConfig(image="python:3.12-slim", network_policy="block"),
            resources=_resources(network_policy="allow"),
        )
    )

    assert fake_client.create_calls[0]["network_policy"] == "block"
    assert fake_client.create_calls[0]["network_mode"] == "none"


@pytest.mark.fast
def test_docker_environment_exec_read_write_upload_download_with_copy_fallback(tmp_path: Path) -> None:
    fake_container = FakeContainer()
    fake_client = FakeDockerClient(fake_container)
    provider = DockerEnvironmentProvider(client=fake_client)
    environment = asyncio.run(_create_environment(provider, DockerEnvironmentConfig(image="python:3.12-slim")))

    result = asyncio.run(
        environment.exec(
            "printf ok",
            cwd="/workspace",
            env={"A": "1"},
            timeout_s=3,
            user="agent",
        )
    )
    assert result.stdout == "ok\n"
    assert fake_container.exec_calls[0]["cwd"] == "/workspace"
    assert fake_container.exec_calls[0]["env"] == {"A": "1"}

    local_upload = tmp_path / "upload.txt"
    local_upload.write_text("uploaded", encoding="utf-8")
    asyncio.run(environment.upload_file(local_upload, "/workspace/upload.txt"))
    assert fake_container.files["/workspace/upload.txt"] == b"uploaded"
    assert fake_container.put_archive_calls[0]["remote_dir"] == "/workspace"

    asyncio.run(environment.write_file("/workspace/written.txt", "written"))
    assert asyncio.run(environment.read_file("/workspace/written.txt")) == b"written"

    local_download = tmp_path / "download.txt"
    asyncio.run(environment.download_file("/workspace/upload.txt", local_download))
    assert local_download.read_text(encoding="utf-8") == "uploaded"
    assert fake_container.get_archive_calls[-1] == "/workspace/upload.txt"

    with pytest.raises(EnvironmentFileNotFoundError):
        asyncio.run(environment.read_file("/workspace/missing.txt"))


@pytest.mark.fast
def test_docker_environment_quotes_paths_used_in_shell_file_probes() -> None:
    class EmptyExecContainer(FakeContainer):
        def exec(self, **kwargs: Any) -> ExecResult:
            self.exec_calls.append(kwargs)
            command = kwargs["command"]
            rendered = command if isinstance(command, str) else " ".join(command)
            return ExecResult(command=rendered, exit_code=0, stdout="", stderr="")

    fake_container = EmptyExecContainer()
    provider = DockerEnvironmentProvider(client=FakeDockerClient(fake_container))
    environment = asyncio.run(_create_environment(provider, DockerEnvironmentConfig(image="python:3.12-slim")))
    remote_path = "/workspace/has spaces; touch /pwned"
    quoted_path = shlex.quote(remote_path)

    asyncio.run(environment.list_files(remote_path))
    assert asyncio.run(environment.is_file(remote_path)) is True
    assert asyncio.run(environment.is_dir(remote_path)) is True

    shell_commands = [call["command"][2] for call in fake_container.exec_calls]
    assert shell_commands[0].startswith(f"find {quoted_path} ")
    assert shell_commands[1] == f"test -f {quoted_path}"
    assert shell_commands[2] == f"test -d {quoted_path}"
    assert f"find {remote_path} " not in shell_commands[0]
    assert f"test -f {remote_path}" not in shell_commands[1]
    assert f"test -d {remote_path}" not in shell_commands[2]


@pytest.mark.fast
def test_docker_environment_creates_nested_parent_before_tar_upload(tmp_path: Path) -> None:
    fake_container = FakeContainer()
    provider = DockerEnvironmentProvider(client=FakeDockerClient(fake_container))
    environment = asyncio.run(_create_environment(provider, DockerEnvironmentConfig(image="python:3.12-slim")))

    asyncio.run(environment.write_file("/workspace/nested dir/file.txt", "payload"))
    local_dir = tmp_path / "local-dir"
    local_dir.mkdir()
    (local_dir / "file.txt").write_text("dir payload", encoding="utf-8")
    asyncio.run(environment.upload_dir(local_dir, "/workspace/nested dir/uploaded"))

    assert fake_container.exec_calls[0]["command"][2] == "mkdir -p '/workspace/nested dir'"
    assert fake_container.put_archive_calls[0]["remote_dir"] == "/workspace/nested dir"
    assert fake_container.exec_calls[1]["command"][2] == "mkdir -p '/workspace/nested dir'"
    assert fake_container.put_archive_calls[1]["remote_dir"] == "/workspace/nested dir"


@pytest.mark.fast
def test_docker_sdk_not_found_maps_to_environment_file_not_found() -> None:
    class NotFound(Exception):
        pass

    class SDKShapedMissingContainer:
        id = "container-sdk"
        name = "container-sdk"

        def get_archive(self, remote_path: str) -> Any:
            raise NotFound(remote_path)

        def stop(self) -> None:
            pass

        def remove(self, *, force: bool = False) -> None:
            del force

    provider = DockerEnvironmentProvider(client=FakeDockerClient(_DockerSDKContainerAdapter(SDKShapedMissingContainer())))
    environment = asyncio.run(_create_environment(provider, DockerEnvironmentConfig(image="python:3.12-slim")))

    with pytest.raises(EnvironmentFileNotFoundError):
        asyncio.run(environment.read_file("/workspace/missing.txt"))


@pytest.mark.fast
def test_docker_provider_respects_mount_optimization(tmp_path: Path) -> None:
    host_workdir = tmp_path / "host-workdir"
    host_workdir.mkdir()
    fake_container = FakeContainer()
    fake_client = FakeDockerClient(fake_container)
    provider = DockerEnvironmentProvider(client=fake_client)
    environment = asyncio.run(
        _create_environment(
            provider,
            DockerEnvironmentConfig(
                image="python:3.12-slim",
                use_host_workdir_mount=True,
                host_workdir=str(host_workdir),
            ),
        )
    )

    local_upload = tmp_path / "mounted-upload.txt"
    local_upload.write_text("mounted upload", encoding="utf-8")
    asyncio.run(environment.upload_file(local_upload, "/workspace/input.txt"))
    assert (host_workdir / "input.txt").read_text(encoding="utf-8") == "mounted upload"
    assert fake_container.put_archive_calls == []

    asyncio.run(environment.write_file("/workspace/nested/output.txt", b"mounted write"))
    assert asyncio.run(environment.read_file("/workspace/nested/output.txt")) == b"mounted write"

    local_download = tmp_path / "mounted-download.txt"
    asyncio.run(environment.download_file("/workspace/nested/output.txt", local_download))
    assert local_download.read_bytes() == b"mounted write"
    assert fake_container.get_archive_calls == []

    description = asyncio.run(environment.describe())
    assert description["capabilities"]["supports_mounts"] is True
    assert description["diagnostics"]["mounted_workdir"] is True
    assert str(host_workdir) not in repr(description)


@pytest.mark.fast
def test_docker_provider_rejects_mounted_workdir_symlink_escape(tmp_path: Path) -> None:
    host_workdir = tmp_path / "host-workdir"
    host_workdir.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("do not expose", encoding="utf-8")
    (host_workdir / "link.txt").symlink_to(secret)

    fake_client = FakeDockerClient()
    provider = DockerEnvironmentProvider(client=fake_client)
    environment = asyncio.run(
        _create_environment(
            provider,
            DockerEnvironmentConfig(
                image="python:3.12-slim",
                use_host_workdir_mount=True,
                host_workdir=str(host_workdir),
            ),
        )
    )

    with pytest.raises(EnvironmentTransferError, match="mounted path contains symlink"):
        asyncio.run(environment.read_file("/workspace/link.txt"))
    with pytest.raises(EnvironmentTransferError, match="mounted path contains symlink"):
        asyncio.run(environment.write_file("/workspace/link.txt", "modified"))

    local_upload = tmp_path / "upload.txt"
    local_upload.write_text("uploaded", encoding="utf-8")
    with pytest.raises(EnvironmentTransferError, match="mounted path contains symlink"):
        asyncio.run(environment.upload_file(local_upload, "/workspace/link.txt"))

    assert secret.read_text(encoding="utf-8") == "do not expose"


@pytest.mark.fast
def test_docker_provider_rejects_mounted_workdir_symlink_component_escape(tmp_path: Path) -> None:
    host_workdir = tmp_path / "host-workdir"
    host_workdir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "file.txt").write_text("outside", encoding="utf-8")
    (host_workdir / "dir_link").symlink_to(outside_dir, target_is_directory=True)

    provider = DockerEnvironmentProvider(client=FakeDockerClient())
    environment = asyncio.run(
        _create_environment(
            provider,
            DockerEnvironmentConfig(
                image="python:3.12-slim",
                use_host_workdir_mount=True,
                host_workdir=str(host_workdir),
            ),
        )
    )

    with pytest.raises(EnvironmentTransferError, match="mounted path contains symlink"):
        asyncio.run(environment.read_file("/workspace/dir_link/file.txt"))
    with pytest.raises(EnvironmentTransferError, match="mounted path contains symlink"):
        asyncio.run(environment.write_file("/workspace/dir_link/file.txt", "modified"))

    assert (outside_dir / "file.txt").read_text(encoding="utf-8") == "outside"


@pytest.mark.fast
def test_docker_provider_rejects_mounted_write_to_symlink_target(tmp_path: Path) -> None:
    host_workdir = tmp_path / "host-workdir"
    host_workdir.mkdir()
    target = host_workdir / "target.txt"
    target.write_text("unchanged", encoding="utf-8")
    (host_workdir / "link.txt").symlink_to(target)

    provider = DockerEnvironmentProvider(client=FakeDockerClient())
    environment = asyncio.run(
        _create_environment(
            provider,
            DockerEnvironmentConfig(
                image="python:3.12-slim",
                use_host_workdir_mount=True,
                host_workdir=str(host_workdir),
            ),
        )
    )

    with pytest.raises(EnvironmentTransferError, match="mounted path contains symlink"):
        asyncio.run(environment.read_file("/workspace/link.txt"))
    with pytest.raises(EnvironmentTransferError, match="mounted path contains symlink"):
        asyncio.run(environment.write_file("/workspace/link.txt", "modified"))

    assert target.read_text(encoding="utf-8") == "unchanged"


@pytest.mark.fast
def test_docker_provider_mounted_transfer_errors_do_not_leak_host_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_workdir = tmp_path / "host-workdir"
    host_workdir.mkdir()
    (host_workdir / "artifact.txt").write_text("payload", encoding="utf-8")

    provider = DockerEnvironmentProvider(client=FakeDockerClient())
    environment = asyncio.run(
        _create_environment(
            provider,
            DockerEnvironmentConfig(
                image="python:3.12-slim",
                use_host_workdir_mount=True,
                host_workdir=str(host_workdir),
            ),
        )
    )

    def raising_copyfile(source: str | Path, target: str | Path) -> None:
        del source, target
        raise OSError(f"permission denied under {host_workdir}")

    monkeypatch.setattr("gage_eval.environment.providers.docker.provider.shutil.copyfile", raising_copyfile)

    with pytest.raises(EnvironmentTransferError) as excinfo:
        asyncio.run(environment.download_file("/workspace/artifact.txt", tmp_path / "download.txt"))

    message = str(excinfo.value)
    assert "/workspace/artifact.txt" in message
    assert str(host_workdir) not in message


@pytest.mark.fast
def test_docker_provider_mounted_read_write_errors_do_not_leak_host_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_workdir = tmp_path / "host-workdir"
    host_workdir.mkdir()
    (host_workdir / "read.txt").write_text("payload", encoding="utf-8")

    provider = DockerEnvironmentProvider(client=FakeDockerClient())
    environment = asyncio.run(
        _create_environment(
            provider,
            DockerEnvironmentConfig(
                image="python:3.12-slim",
                use_host_workdir_mount=True,
                host_workdir=str(host_workdir),
            ),
        )
    )

    def raising_read_bytes(self: Path) -> bytes:
        del self
        raise OSError(f"read failed under {host_workdir}")

    def raising_write_bytes(self: Path, data: bytes) -> int:
        del self, data
        raise OSError(f"write failed under {host_workdir}")

    monkeypatch.setattr(Path, "read_bytes", raising_read_bytes)
    monkeypatch.setattr(Path, "write_bytes", raising_write_bytes)

    with pytest.raises(EnvironmentTransferError) as read_excinfo:
        asyncio.run(environment.read_file("/workspace/read.txt"))
    with pytest.raises(EnvironmentTransferError) as write_excinfo:
        asyncio.run(environment.write_file("/workspace/write.txt", "payload"))

    read_message = str(read_excinfo.value)
    write_message = str(write_excinfo.value)
    assert "/workspace/read.txt" in read_message
    assert "/workspace/write.txt" in write_message
    assert str(host_workdir) not in read_message
    assert str(host_workdir) not in write_message


@pytest.mark.fast
def test_docker_provider_create_maps_client_errors() -> None:
    class RaisingClient(FakeDockerClient):
        def create_container(self, **kwargs: Any) -> FakeContainer:
            raise RuntimeError("docker daemon unavailable")

    provider = DockerEnvironmentProvider(client=RaisingClient())

    with pytest.raises(EnvironmentCreateError, match="docker.create"):
        asyncio.run(_create_environment(provider, {"image": "python:3.12-slim"}))
