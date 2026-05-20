"""Typed metadata facade for summary-affecting run metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gage_eval.reporting.privacy import SecretFilter


RUN_METADATA_SCHEMA_VERSION = 1
_SECRET_FILTER = SecretFilter()


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    run_dir: str
    pipeline_id: str | None = None
    created_at_iso: str | None = None
    config_digest: str | None = None
    summary_generators: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.config_digest and _SECRET_FILTER.redact(self.config_digest).redacted:
            raise ValueError("RunMetadata.config_digest must not contain secrets")

    def to_metadata_fields(self) -> dict[str, Any]:
        run_identity = {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
        }
        if self.pipeline_id:
            run_identity["pipeline_id"] = self.pipeline_id
        if self.created_at_iso:
            run_identity["created_at_iso"] = self.created_at_iso
        fields: dict[str, Any] = {
            "run_metadata_schema_version": RUN_METADATA_SCHEMA_VERSION,
            "run_identity": run_identity,
        }
        if self.config_digest:
            fields["config_digest"] = self.config_digest
        if self.summary_generators:
            fields["summary_generators"] = list(self.summary_generators)
        return fields


@dataclass(frozen=True)
class RuntimeStats:
    sample_count: int
    completed_count: int
    failed_count: int
    aborted_count: int
    scheduler_failed_count: int = 0
    verifier_skipped_count: int = 0
    duration_s: float | None = None
    observability_health: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_runtime_health(
        cls,
        runtime_health: dict[str, Any],
        *,
        observability_health: dict[str, Any] | None = None,
        duration_s: float | None = None,
    ) -> "RuntimeStats":
        return cls(
            sample_count=int(runtime_health.get("sample_count") or 0),
            completed_count=int(runtime_health.get("completed_count") or 0),
            failed_count=int(runtime_health.get("failed_count") or 0),
            aborted_count=int(runtime_health.get("aborted_count") or 0),
            scheduler_failed_count=int(runtime_health.get("scheduler_failed_count") or 0),
            verifier_skipped_count=int(runtime_health.get("verifier_skipped_count") or 0),
            duration_s=duration_s,
            observability_health=dict(observability_health or {}),
        )

    def to_summary_payload(self) -> dict[str, Any]:
        runtime_health = {
            "sample_count": self.sample_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "aborted_count": self.aborted_count,
            "scheduler_failed_count": self.scheduler_failed_count,
            "verifier_skipped_count": self.verifier_skipped_count,
        }
        if self.duration_s is not None:
            runtime_health["duration_s"] = self.duration_s
        payload: dict[str, Any] = {"runtime_health": runtime_health}
        if self.observability_health:
            payload["observability_health"] = dict(self.observability_health)
        return payload


@dataclass(frozen=True)
class ValidationSummary:
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ValidationSummary":
        return cls(dict(payload or {}))

    def to_metadata_value(self) -> dict[str, Any]:
        return dict(_SECRET_FILTER.redact(self.payload).value)
