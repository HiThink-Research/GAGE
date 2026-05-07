"""Strict Docker provider configuration for AgentKit v2 environments."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gage_eval.environment.resources import NetworkPolicy


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=(), strict=True)


DockerMountType = Literal["bind", "volume"]


class DockerMount(_StrictModel):
    source: str
    target: str
    type: DockerMountType = "bind"
    read_only: bool = False

    @field_validator("source", "target")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be non-empty")
        return value

    @field_validator("target")
    @classmethod
    def _target_is_absolute(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("mount target must be absolute")
        return value


class DockerEnvironmentConfig(_StrictModel):
    image: str
    container_name_prefix: str | None = None
    docker_platform: str = "linux/amd64"
    privileged: bool = False
    network_policy: NetworkPolicy | None = None
    mounts: list[DockerMount] = Field(default_factory=list)
    workdir: str = "/workspace"
    exec_workdir: str | None = None
    entrypoint: list[str] | None = Field(default_factory=list)
    keepalive_command: list[str] = Field(default_factory=lambda: ["sleep", "infinity"])
    user: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    use_host_workdir_mount: bool = False
    host_workdir: str | None = None

    @field_validator("image", "docker_platform", "workdir")
    @classmethod
    def _required_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be non-empty")
        return value

    @field_validator("workdir", "exec_workdir")
    @classmethod
    def _workdir_is_absolute(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.startswith("/"):
            raise ValueError("workdir must be absolute")
        return value

    @field_validator("entrypoint", "keepalive_command")
    @classmethod
    def _command_items_must_be_non_empty(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        normalized = [str(item) for item in value]
        if any(not item.strip() for item in normalized):
            raise ValueError("command items must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _validate_mount_optimization(self) -> "DockerEnvironmentConfig":
        if self.use_host_workdir_mount:
            if not self.host_workdir or not self.host_workdir.strip():
                raise ValueError("host_workdir is required when use_host_workdir_mount=true")
            if not self.host_workdir.startswith("/"):
                raise ValueError("host_workdir must be absolute")
        return self


__all__ = ["DockerEnvironmentConfig", "DockerMount", "DockerMountType"]
