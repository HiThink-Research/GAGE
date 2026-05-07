from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from gage_eval.environment import EnvironmentResources
from gage_eval.environment.profiles import EnvironmentProfile
from gage_eval.environment.providers.docker import DockerEnvironmentConfig, DockerEnvironmentProvider


DOCKER_SMOKE_IMAGE = "python:3.12-slim"
DOCKER_SMOKE_PLATFORM = "linux/amd64"


def _docker_client_or_skip() -> Any:
    docker = pytest.importorskip("docker", reason="docker SDK is not installed")
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        pytest.skip(f"Docker daemon is not available: {exc}")
    return client


def _ensure_smoke_image_or_skip(client: Any) -> None:
    try:
        client.images.get(DOCKER_SMOKE_IMAGE)
        return
    except Exception:
        pass
    try:
        client.images.pull(DOCKER_SMOKE_IMAGE, platform=DOCKER_SMOKE_PLATFORM)
    except Exception as exc:
        pytest.skip(f"Docker smoke image is not available and pull failed: {exc}")


async def _create_environment() -> Any:
    provider = DockerEnvironmentProvider()
    return await provider.create(
        kit_id="docker-smoke",
        provider="docker",
        profile_id="docker-smoke-profile",
        profile=EnvironmentProfile(profile_id="docker-smoke-profile", provider="docker", config={}),
        provider_config=DockerEnvironmentConfig(
            image=DOCKER_SMOKE_IMAGE,
            docker_platform=DOCKER_SMOKE_PLATFORM,
            workdir="/workspace",
        ),
        resources=EnvironmentResources(cpu=1.0, memory_gb=1.0, network_policy="block"),
        startup_env={},
        lifecycle="per_sample",
        metadata={"sample_id": "docker-smoke"},
    )


@pytest.mark.io
def test_docker_provider_real_container_exec_transfer_smoke(tmp_path: Path) -> None:
    client = _docker_client_or_skip()
    _ensure_smoke_image_or_skip(client)
    environment = asyncio.run(_create_environment())
    try:
        echo = asyncio.run(environment.exec("echo docker-smoke", timeout_s=10))
        assert echo.exit_code == 0
        assert echo.stdout.strip() == "docker-smoke"

        asyncio.run(environment.write_file("/workspace/smoke.txt", "small payload"))
        assert asyncio.run(environment.read_file("/workspace/smoke.txt")) == b"small payload"

        artifact_path = tmp_path / "artifact.txt"
        asyncio.run(environment.download_file("/workspace/smoke.txt", artifact_path))
        assert artifact_path.read_text(encoding="utf-8") == "small payload"
    finally:
        asyncio.run(environment.stop(delete=True))
