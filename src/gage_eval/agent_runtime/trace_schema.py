from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


TRACE_INLINE_TEXT_LIMIT_BYTES = 4096
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=(), strict=True)


class ArtifactRef(_StrictModel):
    owner: str
    name: str
    path: str
    mime_type: str
    size_bytes: int = Field(ge=0)
    sha256: str

    @field_validator("owner", "name")
    @classmethod
    def _validate_segment(cls, value: str) -> str:
        return _safe_segment(value)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _safe_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("sha256 must be a 64-character lowercase hex digest")
        return value


class TrialResult(_StrictModel):
    trial_id: str
    status: Literal["completed", "failed", "aborted"]
    scheduler_result: dict[str, Any]
    verifier_result: dict[str, Any]
    environment_descriptor: dict[str, Any]
    artifact_refs: list[ArtifactRef]
    trace_ref: ArtifactRef
    failure: dict[str, Any] | None

    @field_validator("trial_id")
    @classmethod
    def _validate_trial_id(cls, value: str) -> str:
        return _safe_segment(value)


class SampleRecord(_StrictModel):
    run_id: str
    task_id: str
    sample_id: str
    dut_id: str
    input_ref: dict[str, Any]
    trial_policy: dict[str, Any]
    trial_results: list[TrialResult]
    aggregate_result: dict[str, Any]
    scheduler_result: dict[str, Any]
    verifier_result: dict[str, Any]
    environment_descriptor: dict[str, Any]
    effective_config_ref: ArtifactRef
    artifacts: list[ArtifactRef]
    status: Literal["completed", "failed", "aborted"]
    failure: dict[str, Any] | None

    @field_validator("run_id", "task_id", "sample_id", "dut_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        return _safe_segment(value)

    @model_validator(mode="after")
    def _reject_known_secrets(self, info: ValidationInfo) -> "SampleRecord":
        secret_values = _secret_values_from_context(info)
        if secret_values and _contains_secret(self.model_dump(mode="python"), secret_values):
            raise ValueError("sample record contains a known secret value")
        return self


class TraceEvent(_StrictModel):
    run_id: str
    task_id: str
    sample_id: str
    trial_id: str
    sequence_no: int = Field(ge=1)
    timestamp: str
    actor: Literal["scheduler", "agent", "environment", "verifier", "runtime"]
    event_type: str
    payload: dict[str, Any]
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)

    @field_validator("run_id", "task_id", "sample_id", "trial_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        return _safe_segment(value)

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be ISO-8601") from exc
        return value

    @model_validator(mode="after")
    def _validate_event_contract(self, info: ValidationInfo) -> "TraceEvent":
        _validate_trace_payload(self.payload)

        if self.event_type == "environment.acquire":
            if self.actor != "environment":
                raise ValueError("environment.acquire events must use actor='environment'")
            if not isinstance(self.payload.get("environment_descriptor"), dict):
                raise ValueError("environment.acquire payload requires environment_descriptor")

        if self.event_type == "verifier.result":
            if self.actor != "verifier":
                raise ValueError("verifier.result events must use actor='verifier'")
            if "metric" not in self.payload:
                raise ValueError("verifier.result payload requires metric")
            if not self.artifact_refs:
                raise ValueError("verifier.result requires artifact_refs")

        secret_values = _secret_values_from_context(info)
        if secret_values and _contains_secret(self.model_dump(mode="python"), secret_values):
            raise ValueError("trace event contains a known secret value")
        return self


def _safe_segment(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("unsafe empty path segment")
    if "\\" in value or "/" in value:
        raise ValueError(f"unsafe path segment: {value!r}")
    if value in {".", ".."} or PurePosixPath(value).is_absolute():
        raise ValueError(f"unsafe path segment: {value!r}")
    if ".." in value:
        raise ValueError(f"unsafe path segment: {value!r}")
    return value


def _safe_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("unsafe empty artifact path")
    if "\\" in value:
        raise ValueError(f"unsafe artifact path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError(f"unsafe artifact path: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe artifact path: {value!r}")
    return value


def _validate_trace_payload(payload: Any) -> None:
    for key, value in _walk_mapping(payload):
        if key in {"stdout", "stderr"} and isinstance(value, str):
            if len(value.encode("utf-8")) > TRACE_INLINE_TEXT_LIMIT_BYTES:
                raise ValueError(f"{key} exceeds inline trace payload limit; write it as an artifact_ref")
        if key in {"patch", "submission_patch"} and isinstance(value, str):
            if _looks_like_full_diff(value):
                raise ValueError(f"{key} must not contain a full diff in trace payload")


def _walk_mapping(value: Any) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            items.append((key_text, child))
            items.extend(_walk_mapping(child))
    elif isinstance(value, list):
        for child in value:
            items.extend(_walk_mapping(child))
    return items


def _looks_like_full_diff(value: str) -> bool:
    return "diff --git " in value or ("\n--- " in value and "\n+++ " in value and "\n@@" in value)


def _secret_values_from_context(info: ValidationInfo) -> tuple[str, ...]:
    context = info.context if isinstance(info.context, dict) else {}
    values = context.get("secret_values", ())
    if isinstance(values, dict):
        return tuple(str(value) for value in values.keys() if value)
    return tuple(str(value) for value in values if value)


def _contains_secret(value: Any, secret_values: tuple[str, ...]) -> bool:
    if isinstance(value, str):
        return any(secret in value for secret in secret_values)
    if isinstance(value, dict):
        return any(_contains_secret(child, secret_values) for child in value.values())
    if isinstance(value, list):
        return any(_contains_secret(child, secret_values) for child in value)
    return False
