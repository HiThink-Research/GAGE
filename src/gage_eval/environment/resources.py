"""Resource and capability records for environment providers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


NetworkPolicy = Literal["allow", "block", "egress_only"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=(), strict=True)


class EnvironmentCapabilities(_StrictModel):
    supports_mounts: bool = False
    supports_upload_download: bool = True
    supports_internet_control: bool = False
    supports_privileged_dind: bool = False
    default_user: str | None = None


class EnvironmentResources(_StrictModel):
    cpu: float | None = Field(default=None, ge=0)
    memory_gb: float | None = Field(default=None, ge=0)
    disk_gb: float | None = Field(default=None, ge=0)
    timeout_s: float | None = Field(default=None, ge=0)
    network_policy: NetworkPolicy = "block"
