"""Shared runtime metadata projection and recording helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.evaluation.cache import EvalCache
from gage_eval.utils.run_identity import RunIdentity, build_run_identity_metadata

RUNTIME_METADATA_SCHEMA_VERSION = 1
RUN_METADATA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RuntimeMetadataSnapshot:
    schema_version: int
    pipeline_id: Optional[str]
    backends: tuple[dict[str, Any], ...]
    agent_runtimes: tuple[dict[str, Any], ...]
    models: tuple[dict[str, Any], ...]
    role_adapters: tuple[dict[str, Any], ...]
    summary_generators: tuple[str, ...]


@dataclass(frozen=True)
class RunMetadataSnapshot:
    schema_version: int
    run_identity: dict[str, Any]


@dataclass(frozen=True)
class SampleRuntimeMetadataSnapshot:
    """Stable sample-scoped runtime metadata written by agent runtimes."""

    session_id: str
    sample_id: str
    scheduler_type: str
    resource_lease: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None


def build_runtime_metadata_snapshot(config: PipelineConfig) -> RuntimeMetadataSnapshot:
    """Project stable runtime metadata from a PipelineConfig."""

    return RuntimeMetadataSnapshot(
        schema_version=RUNTIME_METADATA_SCHEMA_VERSION,
        pipeline_id=config.pipeline_id,
        backends=tuple(
            {
                "backend_id": spec.backend_id,
                "type": spec.type,
                "config": spec.config,
            }
            for spec in (config.backends or [])
        ),
        agent_runtimes=tuple(spec.to_dict() for spec in (config.agent_runtimes or [])),
        models=tuple(
            {
                "model_id": spec.model_id,
                "source": spec.source,
                "hub": spec.hub,
                "hub_params": spec.hub_params,
                "params": spec.params,
            }
            for spec in (config.models or [])
        ),
        role_adapters=tuple(
            {
                "adapter_id": spec.adapter_id,
                "role_type": spec.role_type,
                "backend_id": spec.backend_id,
                "backend_inline": spec.backend,
                "capabilities": list(spec.capabilities or ()),
                "prompt_id": spec.prompt_id,
                "agent_runtime_id": spec.agent_runtime_id,
                "compat_runtime_id": spec.compat_runtime_id,
            }
            for spec in (config.role_adapters or [])
        ),
        summary_generators=tuple(config.summary_generators or ()),
    )


def build_run_metadata_snapshot(identity: RunIdentity) -> RunMetadataSnapshot:
    """Project stable run identity metadata for summaries and diagnostics."""

    return RunMetadataSnapshot(
        schema_version=RUN_METADATA_SCHEMA_VERSION,
        run_identity=build_run_identity_metadata(identity),
    )


def record_runtime_metadata(cache_store: EvalCache, snapshot: RuntimeMetadataSnapshot) -> None:
    """Write runtime metadata using a single stable contract."""

    cache_store.set_metadata("runtime_metadata_schema_version", snapshot.schema_version)
    if snapshot.backends:
        cache_store.set_metadata("backends", [dict(entry) for entry in snapshot.backends])
    if snapshot.agent_runtimes:
        cache_store.set_metadata("agent_runtimes", [dict(entry) for entry in snapshot.agent_runtimes])
    if snapshot.models:
        cache_store.set_metadata("models", [dict(entry) for entry in snapshot.models])
    if snapshot.role_adapters:
        cache_store.set_metadata("role_adapters", [dict(entry) for entry in snapshot.role_adapters])
    if snapshot.summary_generators:
        cache_store.set_metadata("summary_generators", list(snapshot.summary_generators))


def record_run_metadata(cache_store: EvalCache, snapshot: RunMetadataSnapshot) -> None:
    """Write run identity metadata using a single stable contract."""

    cache_store.set_metadata("run_metadata_schema_version", snapshot.schema_version)
    cache_store.set_metadata("run_identity", dict(snapshot.run_identity))


def build_sample_runtime_metadata_snapshot(
    *,
    session_id: str,
    sample_id: str,
    scheduler_type: str,
    resource_lease: dict[str, Any] | None = None,
    failure: dict[str, Any] | None = None,
) -> SampleRuntimeMetadataSnapshot:
    """Project stable sample-scoped runtime metadata."""

    return SampleRuntimeMetadataSnapshot(
        session_id=session_id,
        sample_id=sample_id,
        scheduler_type=scheduler_type,
        resource_lease=dict(resource_lease or {}) or None,
        failure=dict(failure or {}) or None,
    )
